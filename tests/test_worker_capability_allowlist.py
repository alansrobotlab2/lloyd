"""Every worker source's reach comes from a declared allow-list, not a deny union (#2269).

A worker turn has always been refused by a real gate: `_pre_dispatch` checks
`options.disallowed_tools`, then `options.allowed_tools`, before any hook runs.
What was missing was the policy behind the value. Each source concatenated
config's `disabled_tools` + `WORKER_AUTOMOD_BAN` + `WORKER_GRANT_MINT_BAN` + its
own `DISALLOWED`, so the reachable set was "everything nobody thought to name" —
and the thing nobody thought to name, on the turn whose whole input is a
transcript fetched off the internet, turned out to be the Thunderbird senders
and the calendar and contacts mutators. Measured against the live 147-tool pool
on 2026-10-06: `youtube-digest` reached 99 tools, 12 of them tier-2/3.

The fixture these tests grade against is
`tests/fixtures/worker_capability_baseline.json` — the step-1 table, before and
after per source, regenerated with
`LLOYD_RECORD_CAPABILITY_BASELINE=1 pytest tests/test_worker_capability_allowlist.py`.
A test recomputes it from the live compile on every run, so the artifact cannot
rot into a description of code that no longer exists.

Two rules about how the refusals below are asserted, because an allow-list is
easy to test vacuously:

* Nothing here monkeypatches `_pre_dispatch` or spies on it. Every dispatch
  claim runs the real function with the source's real `RunOptions`, the same way
  `tests/test_harness_disallowed_tools.py` does — a spy would verify the wiring
  and miss the gate (the review rung's finding on #1136).
* Every refusal is asserted to be *refused by the envelope*, by its own message
  (`"not available on this turn"`), and every allowance is paired with a control
  in the same shape on a turn that has no envelope. A tool that is refused
  because of an unrelated rail, or granted because nothing on this box happened
  to object, is not this file's subject.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

import workers.sources.autotriage as autotriage
import workers.sources.deep_research as deep_research
import workers.sources.session_distill as session_distill
import workers.sources.youtube_digest as youtube_digest
from app.harness import guard_arm_matrix as gam
from app.harness import capabilities as caps
from app.harness.capabilities import UnknownCapability
from app.harness.hooks import HookRegistry
from app.harness.loop import _allow_list_hidden, _allowed_names, _pre_dispatch
from app.harness.options import RunOptions
from app.harness.policy import TIER2_TOOLS, TIER3_TOOLS, normalize_tool_name
from app.harness.tool_roster import REGISTERED_TOOLS
from app.harness.tool_search import LoadedToolSet
from workers.sources import _common as C

FIXTURE = Path(__file__).parent / "fixtures" / "worker_capability_baseline.json"

#: The sources the item names as untrusted-ingest: their turn's whole input is
#: text somebody else wrote. Named here as well as in `capabilities` so that
#: widening the fleet's most-protected list needs an edit in two files.
UNTRUSTED = ("session-distill", "youtube-digest", "deep-research")

#: The three names the acceptance clause singles out, one per family, each a
#: tool whose whole job is mutating something outside this machine's filesystem.
SINGLE_REFUSALS = ("email_send", "contacts_delete", "calendar_delete_event")

#: The source #2671 declares. It is NOT an untrusted-ingest source — its turn's
#: input is Lloyd's own backlog, which this fleet already trusts enough to
#: implement from — so it joins `UNTRUSTED` and
#: `capabilities.UNTRUSTED_INGEST_SOURCES` nowhere here: widening the fleet's
#: most-protected list is its own round's ruling, and the row-count claim that
#: moves with this one is the guard matrix's, pinned in
#: `tests/test_guard_arm_matrix.py`.
TRIAGE_SOURCE = "autotriage"

#: What each source's own job writes. `youtube-digest`'s are the two its prompt
#: names (`Write`/`vault_write` for the note, `fact_add` for the entity);
#: `session-distill`'s is the fact write the distiller exists to make, plus
#: `Bash`, which is how it reads the transcripts it distils (344 of the 560 tool
#: calls in its 54 retained sessions, every one in the last fortnight).
#: `autotriage`'s three (#2671) are the whole triage loop's output surface:
#: `backlog_write_task` (the verdict append and the one item it may file, in
#: 1176 of its 1734 retained sessions), `Read` (455 files) and `Bash` (1714 —
#: the one cheap check that decides a premise). A source in this table without
#: a declared set is a node that cannot pass, which is the point: the table is
#: the census of what a declaration must never strip. The node passes for an
#: undeclared source too, so its non-vacuity is the paired refusal node on the
#: same turn — `test_autotriage_base_refuses_every_durable_external_name_by_name`.
JOBS_MUST_STILL_WORK = {
    "session-distill": ("fact_add", "Bash", "Read"),
    "youtube-digest": ("vault_write", "Write", "fact_add", "backlog_write_task"),
    "deep-research": ("vault_write", "fact_add", "http_fetch"),
    TRIAGE_SOURCE: ("backlog_write_task", "Read", "Bash"),
}

#: Every source the dispatch node below covers, read back out of the table so a
#: key can be added to the census without joining the parameterization
#: (#2671's triage finding: the node was parameterized over `UNTRUSTED` alone,
#: so declaring a fourth source pinned nothing about its own writes surviving).
JOB_TABLE_SOURCES = tuple(sorted(JOBS_MUST_STILL_WORK))

SOURCE_MODULES = {
    "session-distill": session_distill,
    "youtube-digest": youtube_digest,
    "deep-research": deep_research,
    # #2671. `autotriage` has no module-level `DISALLOWED`: its single-item and
    # group paths pass no `extra_disallowed` at all, so the `getattr` below
    # compiles exactly what those two turn shapes ship. Its third path narrows
    # by `SWEEP_DISALLOWED`, which is its own node below rather than a value
    # folded into the shared table — the artifact measures a source's base.
    "autotriage": autotriage,
}

#: The pool as a worker would see it: the committed roster, which is what
#: `capabilities` validates declared names against. Discovery is not run by any
#: test here, so this is the same set the boot-time check used.
POOL = frozenset(REGISTERED_TOOLS)


def _options(source: str) -> RunOptions:
    """The turn options that source's own call site would build.

    Its real `extra_disallowed` — the `DISALLOWED` constant from its own module,
    which is what `youtube-digest.py:770` passes — so the envelope under test is
    the one that ships, not one assembled from the test's own idea of the job.
    """
    mod = SOURCE_MODULES[source]
    return C._worker_run_options(
        4, source=source,
        extra_disallowed=list(getattr(mod, "DISALLOWED", ()) or ()))


def _reach_before(opts: RunOptions) -> frozenset[str]:
    """What the deny list alone left reachable — the pre-#2269 view.

    Read from the same options object, not from a remembered list: the deny
    compile is unchanged by this work, so a source's "before" column and its
    "after" column come from one build and cannot be describing different code.
    """
    denied = {normalize_tool_name(d) for d in (opts.disallowed_tools or [])}
    return POOL - denied


def _reach_after(opts: RunOptions) -> frozenset[str]:
    """What the harness will actually let through, envelope included."""
    reach = _reach_before(opts)
    allowed = _allowed_names(opts)
    return reach if allowed is None else (reach & allowed)


def _tc(name: str, args: dict | None = None) -> dict:
    return {"id": "call_1", "function": {"name": name},
            "_args_dict": args or {}}


def _dispatch(opts: RunOptions, name: str, args: dict | None = None):
    """The real pre-dispatch gate, with nothing replaced.

    A `None` return means the call went through to the MCP layer, which is what
    "still dispatches" means here; a `NormalizedEvent` means it was refused
    before the pool saw it.
    """
    return asyncio.run(_pre_dispatch(
        tc=_tc(name, args), options=opts, session_id="s",
        loaded_set=LoadedToolSet(enabled=False, catalog=[], loaded=set())))


def _table() -> dict:
    """The before/after table, recomputed. The artifact, and its own check."""
    rows: dict[str, dict] = {}
    for source in sorted(caps.SOURCE_CAPABILITIES) + sorted(
            set(gam.worker_source_names()) - set(caps.SOURCE_CAPABILITIES)):
        if source in SOURCE_MODULES:
            opts = _options(source)
        else:
            opts = C._worker_run_options(4, source=source)
        before, after = _reach_before(opts), _reach_after(opts)
        rows[source] = {
            "declared": opts.allowed_tools is not None,
            "n_allowed": None if opts.allowed_tools is None
            else len(opts.allowed_tools),
            "reachable_before": len(before),
            "reachable_after": len(after),
            "durable_external_before": sorted(caps.durable_external(before)),
            "durable_external_after": sorted(caps.durable_external(after)),
        }
    return {"pool": len(POOL), "sources": rows}


# ── The step-1 artifact ─────────────────────────────────────────────────────

@pytest.mark.skipif(not os.environ.get("LLOYD_RECORD_CAPABILITY_BASELINE"),
                    reason="recording mode only (LLOYD_RECORD_CAPABILITY_BASELINE=1)")
def test_record_the_baseline():
    FIXTURE.write_text(json.dumps(_table(), indent=1, sort_keys=True) + "\n")


def test_the_baseline_artifact_is_what_the_code_compiles_today():
    """The table is a measurement, so it is re-measured — never re-quoted.

    This is the node that keeps `worker_capability_baseline.json` an artifact
    rather than a comment: it compares every cell against a fresh compile, so
    widening one source's set without regenerating the file goes red here,
    naming the source.
    """
    recorded = json.loads(FIXTURE.read_text())
    live = _table()
    assert recorded == live, (
        "the committed baseline no longer describes the compile; regenerate "
        "with LLOYD_RECORD_CAPABILITY_BASELINE=1 and read the diff")
    assert recorded["pool"] >= 100, (
        f"a {recorded['pool']}-tool pool is not the fleet's pool; the roster "
        "snapshot is stale or discovery never ran")
    # A zero in the after column means nothing unless the same census finds the
    # name when it is reachable. The tier ladder is the census's own definition,
    # so it is checked against the tools the clause names, and then against the
    # before column of the same rows — one grep and two measurements.
    ladder = set(TIER2_TOOLS) | set(TIER3_TOOLS)
    assert {"email_send", "contacts_delete", "calendar_delete_event"} <= ladder
    for source in UNTRUSTED:
        assert "email_send" in recorded["sources"][source]["durable_external_before"], (
            f"{source}'s before column shows no durable-external reach at all, "
            "so the after column's zero is a blind census, not a fix")


def test_every_untrusted_ingest_source_reaches_no_durable_external_tool():
    """The acceptance measurement, over the table's own after column."""
    rows = _table()["sources"]
    for source in UNTRUSTED:
        row = rows[source]
        assert row["declared"], f"{source} has no declared capability set"
        assert row["durable_external_after"] == [], (
            f"{source} reaches {row['durable_external_after']} after the "
            "envelope; the row's before column names what it used to reach")


# ── Clause 1: the envelope is declared, compiled, and advertised ────────────

@pytest.mark.parametrize("source", UNTRUSTED)
def test_a_declared_source_gets_an_allow_list_and_keeps_the_deny_floor(source):
    """`allowed_tools` is non-None, and the standing bans are still in the list.

    Both halves matter. `allowed_tools` alone would be the whole clause if the
    deny compile were safe to drop — it is not: `disallowed_tools` is what the
    per-iteration refresher re-derives when plan mode flips mid-turn, and the
    automod and grant-mint bans are enforced twice on purpose.
    """
    opts = _options(source)
    assert opts.allowed_tools is not None, f"{source} still runs on a deny union"
    assert set(opts.allowed_tools) == set(caps.expand(
        caps.SOURCE_CAPABILITIES[source]))
    banned = {normalize_tool_name(t) for t in C.WORKER_AUTOMOD_BAN}
    denied = {normalize_tool_name(d) for d in opts.disallowed_tools}
    assert banned <= denied, "the automod ban left the deny compile"
    assert not (set(opts.allowed_tools) & banned), (
        "a tool the standing ban refuses must not sit in a declared envelope")


@pytest.mark.parametrize("source", UNTRUSTED)
def test_the_advertised_catalog_is_exactly_the_declared_envelope(source):
    """What the turn is TOLD it has, computed by the loop's own hiding pass.

    Not a second implementation of the set-difference this feature does:
    `_allow_list_hidden` is the function `run_turn` calls to build the wire
    `tools=` array, so this asserts the advertised set against the mechanism
    that ships it rather than against a copy of its logic.
    """
    opts = _options(source)
    # `pool.discovered` is `[(server_name, [ {name:...}, ... ])]`, which
    # is the shape `_allow_list_hidden` unpacks; the fixture is its
    # own, not a flat list, so the node exercises the real join.
    discovered = [("lloyd-mcp", [{"name": n} for n in sorted(POOL)])]
    hidden = _allow_list_hidden(opts, discovered)
    assert hidden == {n for n in POOL if n not in set(opts.allowed_tools)}
    # And the catalog that survives the hiding is the envelope, tool for tool.
    assert POOL - hidden == set(opts.allowed_tools)
    # And a turn with no envelope hides nothing on this axis, which
    # is what makes the assertion above a measurement of the
    # envelope rather than of the fixture.
    assert _allow_list_hidden(
        RunOptions(model="primary"), discovered) == set()


@pytest.mark.parametrize("source", UNTRUSTED)
def test_the_block_shipped_to_the_turn_names_the_same_envelope(source):
    """`build_denied_tools_block` reads the final value, not an earlier draft.

    The item's out-of-scope rule is that this change adds no second enforcement
    layer; the mirror of that is that the prompt must state the value dispatch
    enforces. A block naming only the deny list on an allow-listed turn is a
    turn told "57 things are refused" with no answer to what remains, which is
    the shape that makes a model retry a tool it never had.
    """
    opts = _options(source)
    block = C.build_denied_tools_block(opts.disallowed_tools,
                                       allowed=opts.allowed_tools)
    assert C.CAPABILITY_HEADING in block
    stated = block.split(C.CAPABILITY_HEADING, 1)[1].split("Else", 1)[0]
    for name in opts.allowed_tools:
        assert f" {name}" in stated or stated.lstrip().startswith(name), (
            f"{name} is enforced but not stated")
    assert C.DENIED_TOOLS_HEADING in block, (
        "the named refusals #1066 exists for are gone: a worker that learns a "
        "tool is gone from the RESULT block retried it 22 times in 4 days")
    # The prompt must not name a tool the envelope grants as refused: the two
    # sentences have to be consistent, so the deny sentence may only list names
    # the envelope also refuses.
    denied_block = block.split(C.DENIED_TOOLS_HEADING, 1)[1].split(
        "\n", 1)[0].rstrip(".")
    named = {n.strip() for n in denied_block.split(": ", 1)[1].split(",")}
    assert not ({normalize_tool_name(n) for n in named}
                & set(opts.allowed_tools))


# ── Clause 2: the three refusals, at dispatch ───────────────────────────────

@pytest.mark.parametrize("source", UNTRUSTED)
@pytest.mark.parametrize("tool", SINGLE_REFUSALS)
def test_the_acceptance_refusals_are_refused_at_dispatch(source, tool):
    """Through the real `_pre_dispatch`, on that source's own options.

    The paired control is `test_a_durable_external_tool_still_dispatches_for_a_
    source_that_declared_it`: the same tool, the same call path, a turn with no
    envelope — there it reaches the pool. Without that pair this node would pass
    on a box where the tool simply did not exist.
    """
    opts = _options(source)
    evt = _dispatch(opts, tool, {"to": "a@b.c", "subject": "s", "body": "x"})
    assert evt is not None, f"{tool} dispatched on a {source} turn"
    assert evt["is_error"] is True
    assert "not available on this turn" in evt["content"], (
        f"refused, but not by the envelope: {evt['content'][:120]}")


def test_a_durable_external_tool_still_dispatches_for_a_source_that_declared_it():
    """The control that makes the refusal above mean something.

    Same function, same tool name, no envelope: the pool is a live MCP server
    for these tools, so `_pre_dispatch` returns None and the call proceeds. What
    the envelope did is the difference between the two nodes, and nothing else.
    """
    plain = RunOptions(model="primary", disallowed_tools=list(C.WORKER_AUTOMOD_BAN),
                       hooks=HookRegistry())
    for tool in SINGLE_REFUSALS:
        assert _dispatch(plain, tool) is None, (
            f"{tool} was refused with no envelope — some other rail is firing, "
            "and the refusal node above is measuring the wrong thing")


def test_the_envelope_refuses_the_legacy_spelling_too():
    """`mcp__lloyd-mcp__email_send` is the same call (#727's route).

    The allow-list is checked on the normalised name precisely so the legacy
    spelling is not a way past it; this is the envelope, on a worker turn, and
    an untrusted-ingest source — the case #727's normalisation was written for
    but never had an allow-list behind it.
    """
    opts = _options("youtube-digest")
    evt = _dispatch(opts, "mcp__lloyd-mcp__email_send")
    assert evt is not None and "not available on this turn" in evt["content"]


# ── Clause 3: the job's own writes still go through ─────────────────────────

@pytest.mark.parametrize("source", JOB_TABLE_SOURCES)
def test_each_source_still_dispatches_the_writes_its_own_job_does(source):
    """The whole failure mode of an allow-list is a job that quietly lost a tool.

    Each name here is one the source's own prompt instructs the turn to call or
    its retained session transcripts show it calling — see the note in
    `app/harness/capabilities.py`. A `None` from `_pre_dispatch` means the call
    reached the MCP layer, which is the only claim this node makes: the grant
    gate, the safety tier of a particular `Bash` command and the content gate
    are other layers with their own suites.

    The parameterization is the table's own key set, not `UNTRUSTED`: this is
    the node #2671's triage named as the gap a declaration can hide behind,
    because a source that is declared but absent from here pins nothing about
    its writes surviving, and for `autotriage` the symptom of that is the
    triage loop going quiet rather than an error. The control that this is not
    a vacuous pass is the paired refusal node on the same turn
    (`test_autotriage_base_refuses_every_durable_external_name_by_name`).
    """
    opts = _options(source)
    for tool in JOBS_MUST_STILL_WORK[source]:
        evt = _dispatch(opts, tool, {"command": "ls"} if tool == "Bash" else None)
        assert evt is None or not (
            evt.get("is_error") and "not available on this turn" in evt["content"]), (
            f"{source}'s own write {tool!r} is outside its envelope: "
            f"{(evt or {}).get('content', '')[:120]}")


# ── Clause 4: a typo cannot silently shrink a source ────────────────────────

def test_an_unknown_declared_name_raises_at_import_not_at_dispatch():
    """The failure mode only an allow-list has, closed at boot.

    With a deny list a typo is harmless — an extra name nobody can call. With an
    allow-list `vault_wriet` means the note never gets written, and the symptom
    a human sees is a job that "just stopped working". So the name is checked
    against the pool here, at import, and the exception names it.
    """
    with pytest.raises(UnknownCapability) as exc:
        caps.expand(("READ_FILES", "vault_wriet"))
    assert "vault_wriet" in str(exc.value)
    # And a standing ban cannot be smuggled in as a capability.
    with pytest.raises(UnknownCapability):
        caps.expand(("automod_start",))


def test_the_shipped_declarations_all_resolve(tmp_path, monkeypatch):
    """Re-importing the module with the shipped sets must not raise.

    `_validate_declarations()` runs at import; this drives that path a second
    time so a set that only *looks* well-formed (a name valid in one expand call
    and not another) cannot survive.
    """
    import importlib

    importlib.reload(caps)
    for source, declared in caps.SOURCE_CAPABILITIES.items():
        assert caps.expand(declared) <= POOL or source == "no-such"
        assert caps.envelope_for(source) is not None


def test_a_source_that_declares_nothing_keeps_todays_compile():
    """The strict default is today's behaviour, not the pool and not an empty set.

    Eleven sources still run on the deny union, one source per round is this
    item's own risk rule, and the alternative — defaulting an undeclared source
    to nothing — would take the fleet down on the day this landed. Every one of
    them still appears in the table, which is how the owed work stays visible.
    """
    undeclared = sorted(set(gam.worker_source_names())
                        - set(caps.SOURCE_CAPABILITIES))
    assert len(undeclared) >= 10, (
        f"only {len(undeclared)} sources are undeclared; the table's owed "
        "column has quietly emptied")
    for source in undeclared:
        opts = C._worker_run_options(4, source=source)
        assert opts.allowed_tools is None
        assert _allow_list_hidden(
        opts, [("lloyd-mcp", [{"name": n} for n in sorted(POOL)])]) == set()
    rows = _table()["sources"]
    for source in undeclared:
        assert rows[source]["reachable_after"] == rows[source]["reachable_before"]


# ── The seam: worker pool → backend, two processes, one envelope ────────────

def _post_body(source: str, extra: list[str], monkeypatch) -> dict:
    """The JSON body `run_prompt_in_session` posts for a source, at the transport.

    `httpx.AsyncClient` is the socket, not the mechanism under test: what is
    being checked is that the pool's declared envelope is *in the bytes* the
    backend reads. The reply is a single `done` event so the SSE reader exits the
    way it does on a finished turn.

    Every session-backed source goes through this one function — the two ingest
    sources from their own `_stream`, and `autotriage` from all three of its turn
    shapes (#2671) — so this one capture covers all of them.
    """
    import httpx

    captured: dict = {}

    class _Resp:
        status_code = 200

        async def aread(self):
            return b"{}"

        async def aiter_lines(self):
            for line in ('event: done',
                         'data: {"response": "ok", "stop_reason": "end_turn"}',
                         ''):
                yield line

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, json=None, headers=None):
            captured.update(json)

            class _Ctx:
                async def __aenter__(self_inner):
                    return _Resp()

                async def __aexit__(self_inner, *a):
                    return False

            return _Ctx()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    asyncio.run(C.run_prompt_in_session(
        "p", title="t", source=source, max_turns=2, timeout_seconds=30,
        extra_disallowed=extra, session_id="s1", inner_voice=False))
    return captured


def test_the_loopback_post_carries_the_envelope_the_backend_applies(monkeypatch):
    """One seam, both ends, in one test.

    `youtube-digest`, `deep-research` and `autotriage` do not build their options
    in the pool: they POST to the backend, which builds them in another process
    from this body and nothing else — the same seam `grant_scope` (#534) and
    `effect_scope` (#544) had to be crossed on their own, for the same reason.
    So a change that stopped at `_worker_run_options` would have shipped an
    envelope for the sources that read a bundle off disk and none for the ones
    that read the internet or the board. #2671 adds `autotriage` to this loop
    because that is where its ceiling actually binds: all three of its turn
    paths (`execute` single-item, `_execute_group`, `_execute_sweep`) go through
    `run_prompt_in_session` with `source=NAME`, so a base that never crossed the
    POST would leave 1615 runs a week on the deny union while every in-process
    node above stayed green. This node takes the body the real `_stream` builds
    and hands it to the real `build_turn_options`, which is the only thing
    standing between the pool's declaration and the turn that runs.
    """
    import json as _json

    from app.routers import messages as M
    from app.routers import turn_options as topts

    for source in ("autotriage", "youtube-digest", "deep-research"):
        extra = list(getattr(SOURCE_MODULES[source], "DISALLOWED", ()))
        body = _post_body(source, extra, monkeypatch)
        assert body.get("allowed_tools"), (
            f"{source}'s POST carries no envelope: the backend would rebuild "
            "the turn from the deny union alone")
        assert body["allowed_tools"] == _options(source).allowed_tools, (
            f"{source}'s two turn shapes disagree")

        # The far end. A worker session file, then the real builder.
        sid = f"cap-{source}"
        (monkeypatchsessions(monkeypatch, M) / f"{sid}.json").write_text(_json.dumps(
            {"session_id": sid, "platform": "worker", "source": source,
             "messages": []}))
        snapshot = topts.SessionSnapshot.load(sid, M.SESSIONS_DIR)
        opts = topts.build_turn_options(
            snapshot, {"session_id": sid, "allowed_tools": body["allowed_tools"]},
            "stream", text="p").options
        assert opts.allowed_tools == body["allowed_tools"], (
            "the endpoint did not apply the envelope that arrived in its body")
        assert set(opts.allowed_tools) == set(caps.expand(
            caps.SOURCE_CAPABILITIES[source]))


def monkeypatchsessions(monkeypatch, M):
    """Point the router's session dir at scratch, as the grant-gate tests do."""
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    monkeypatch.setattr(M, "SESSIONS_DIR", d)
    return d


def test_a_chat_turn_that_sends_the_key_keeps_every_tool(monkeypatch):
    """The break-glass, tested from the attacker's side.

    The key is honoured on the worker surface and ignored everywhere else, so a
    request that names an interactive session and sends `allowed_tools` cannot
    clip a person's own turn — which is the shape of the incident this feature
    would otherwise introduce. A chat session file, the same body, the
    opposite expectation.
    """
    import json as _json

    from app.routers import messages as M
    from app.routers import turn_options as topts

    sid = "cap-chat"
    monkeypatchsessions(monkeypatch, M)
    (M.SESSIONS_DIR / f"{sid}.json").write_text(_json.dumps(
        {"session_id": sid, "platform": "mission-control", "messages": []}))
    snapshot = topts.SessionSnapshot.load(sid, M.SESSIONS_DIR)
    opts = topts.build_turn_options(
        snapshot, {"session_id": sid,
                   "allowed_tools": ["Read"]}, "stream", text="p").options
    assert opts.allowed_tools is None, (
        "a non-worker turn honoured an envelope: one POST could now reduce a "
        "person's chat to one tool")


# ── Clause 5: the guard matrix consumes the declarations ────────────────────

def test_the_guard_matrix_reports_a_source_reaching_beyond_its_declaration():
    """Drift is reported the way an unarmed registry is (#1963's mechanism).

    A new source that reaches a durable-external tool appears in the same report
    that names an un-armed guard, because the denominator is the roster's own
    import list rather than a hand-kept table.
    """
    m = gam.capability_matrix()
    names = gam.worker_source_names()
    assert names == set(m), "the capability report and the source roster differ"
    rows = gam.capability_findings()
    undeclared = {f.target for f in rows
                  if f.reason == gam.FINDING_UNDECLARED_ENVELOPE}
    assert undeclared == names - set(caps.SOURCE_CAPABILITIES)
    for source in UNTRUSTED:
        assert m[source]["untrusted_ingest"] is True
        assert m[source]["durable_reach"] == []


def test_an_ingest_set_that_grants_a_sender_is_reported(monkeypatch):
    """The negative case, forged rather than hypothetical.

    Without this node the only evidence that the finding exists is that no
    source trips it — which is also true of a check that can never fire.
    """
    forged = dict(caps.SOURCE_CAPABILITIES)
    forged["youtube-digest"] = tuple(forged["youtube-digest"]) + ("email_send",)
    monkeypatch.setattr(caps, "SOURCE_CAPABILITIES", forged)
    found = gam.capability_findings()
    assert [f for f in found
            if f.reason == gam.FINDING_INGEST_REACHES_DURABLE
            and f.target == "youtube-digest"]


# ── #2471: scheduled-task's declared ceiling and its per-job grant widening ──
#
# Everything below compiles the source the way the fleet's single compile point
# does — `C._worker_run_options(4, source="scheduled-task")` — which is the row
# `tests/fixtures/worker_capability_baseline.json` measures and the guard
# matrix reports. The census numbers quoted here were taken on 2026-10-09 over
# `~/lloyd-data/sessions/*_autonomy_*.json`: 566 files, so the item's >5% rule
# is 29 files.
#
# The grant rows these nodes need are minted into a fixture-scoped store,
# never the live `workers.db`: `authority_grants` holds exactly one row (id 1,
# scope `autonomy-task:40`), and it was revoked on 2026-10-07 — before this
# item was filed. "A live grant exists" is a state this box does not have, so
# the test builds it and resets the contextvar it binds.

import app.harness.policy as policy_mod  # noqa: E402  (after the constants it patches)

SCHED_SOURCE = "scheduled-task"
GRANT_NAME = "autonomy_write_task"   # the item's widening example: tier-2, grant-only
THIS_SCOPE = "autonomy-task:77"
OTHER_SCOPE = "autonomy-task:78"
_FAR_FUTURE = "2099-01-01T00:00:00+00:00"


def _sched_opts():
    """The scheduled-task turn as the pool's one compile point builds it."""
    return C._worker_run_options(4, source=SCHED_SOURCE)


def _fixture_grant_store(tmp_path):
    """A `GrantStore` on a scratch file — the live `workers.db` is read by no node here."""
    from app.harness.policy import GrantStore
    store = GrantStore(tmp_path / "grants.db")
    store.ensure_schema()
    return store


def _bind_scope(request, scope: str):
    """Bind the pool's per-job scope contextvar, restored at teardown.

    A token/reset pair rather than a contextvar-`setattr`: the leak this
    avoids is the one that would matter — a test left with a bound
    `autonomy-task:<id>` scope would make the next test's baseline compile a
    widened one, and the artifact-vs-live comparison node would go red on the
    wrong row.
    """
    token = policy_mod.current_scope.set(scope)
    request.addfinalizer(lambda: policy_mod.current_scope.reset(token))


def _use_store(monkeypatch, store):
    monkeypatch.setattr(policy_mod, "default_store", lambda: store)


def test_scheduled_task_gets_a_declared_ceiling_smaller_than_the_deny_union():
    """Clause 1: the ceiling exists, it shrinks the reach, and the matrix row is gone.

    `reachable_after == allowed_tools` is the identity worth pinning, not just
    the inequality: it says the declared set IS the reach — no name the turn
    holds is refused by an unrelated rail, and no name outside the ceiling
    slips in beside it. The matrix half names the source rather than counting
    rows, so the next source declared does not move this node (#2471's triage
    finding on `test_guard_arm_matrix.py:257`'s `>= 10`).
    """
    opts = _sched_opts()
    assert opts.allowed_tools is not None, "scheduled-task still runs on a deny union"
    before, after = _reach_before(opts), _reach_after(opts)
    assert len(after) < len(before), (
        f"the declared ceiling reaches {len(after)} of the {len(before)} "
        "names the deny union left — it is not a ceiling")
    assert set(after) == set(opts.allowed_tools)
    undeclared = {f.target for f in gam.capability_findings()
                  if f.reason == gam.FINDING_UNDECLARED_ENVELOPE}
    assert SCHED_SOURCE not in undeclared, (
        "the guard matrix still prints a worker-source-declares-no-capability-set "
        "row for scheduled-task")


def test_the_baseline_row_for_scheduled_task_reads_declared_with_no_durable_external():
    """Clause 2: the committed artifact describes the new compile.

    `n_allowed == reachable_after` on the artifact is the same identity the
    node above measures live; pinned on the FILE, it fails if someone records
    a table from a tree where a grant row was live (widening must not leak
    into the committed baseline — the recording compile runs unscoped).
    `durable_external_after == []` is the item's "26 become a base set plus a
    declared widening": the base reaches none of the 26 by its own names.
    """
    recorded = json.loads(FIXTURE.read_text())
    row = recorded["sources"][SCHED_SOURCE]
    assert row["declared"] is True
    assert row["n_allowed"] == row["reachable_after"], (
        "the baseline was recorded with a widening live, or against a pool "
        "that does not serve every allowed name")
    assert row["durable_external_after"] == []
    assert row["reachable_after"] < row["reachable_before"]


def test_every_base_capability_still_dispatches_on_a_scheduled_task_turn():
    """Clause 3: nothing in the base set dies at the real gate.

    This is the node that makes the base set a census and not a wish: it
    iterates the shipped ceiling itself, not a hand-picked list, so a future
    edit that declares a name `_pre_dispatch` refuses for some reason beside
    the envelope (a tier the grant gate answers, an arg the safety ladder
    reads) goes red here naming the name. `Bash` carries a benign command
    because its tier is the command string (`safety.bash_command_tier`).
    """
    opts = _sched_opts()
    assert opts.allowed_tools
    for name in opts.allowed_tools:
        evt = _dispatch(opts, name, {"command": "ls"} if name == "Bash" else None)
        assert evt is None, (
            f"base capability {name!r} did not reach the MCP layer: "
            f"{(evt or {}).get('content', '')[:160]}")


def test_a_live_grant_widens_this_job_and_a_revoked_one_refuses(tmp_path, monkeypatch, request):
    """Clause 4, both edges: advertised + dispatched while live, refused once revoked.

    The live edge runs the real grant gate too — `_pre_dispatch` on the
    source's own hooks, under the bound `autonomy-task:77` scope, with the
    row live — so `None` means both the envelope widened AND the tier-2
    grant check answered for it, the two layers agreeing. The revoked edge
    asserts the refusal is the envelope's own (`not available on this turn`),
    the same message rule the untrusted-ingest refusals use above.
    """
    store = _fixture_grant_store(tmp_path)
    row = store.mint(scope=THIS_SCOPE, tool_pattern=GRANT_NAME,
                     issued_by="test-#2471", expires_at=_FAR_FUTURE,
                     note="clause-4 fixture: this job's own widening")
    _use_store(monkeypatch, store)
    _bind_scope(request, THIS_SCOPE)

    opts = _sched_opts()
    assert GRANT_NAME in opts.allowed_tools, (
        "the live grant for this job's own scope did not widen the ceiling")
    discovered = [("lloyd-mcp", [{"name": n} for n in sorted(POOL)])]
    assert GRANT_NAME not in _allow_list_hidden(opts, discovered), (
        "widened but not advertised: the turn was never told it holds the tool")
    assert _dispatch(opts, GRANT_NAME, {"id": 68}) is None, (
        "the widened name does not dispatch on this turn's own options")

    assert store.revoke(row["id"]) is True
    opts2 = _sched_opts()
    assert GRANT_NAME not in opts2.allowed_tools, (
        "a revoked grant still widens: live() read it as alive")
    evt = _dispatch(opts2, GRANT_NAME, {"id": 68})
    assert evt is not None and "not available on this turn" in evt["content"], (
        f"revoked edge refused by the wrong rail: "
        f"{(evt or {}).get('content', '')[:160]}")


def test_widening_is_per_job_and_cannot_smuggle(tmp_path, monkeypatch, request):
    """Clause 5: another job's grant, a standing ban and an unserved name all stay out.

    The three negatives mint against the same store and scope this turn has —
    only their own properties exclude them: a different `scope` (widening is
    per job, never across jobs), membership in the standing bans
    (`app.tool_bans`; an allow-list that "grants" `automod_start` must not
    survive, and `expand` raises for the same case at import), and a name no
    server serves. The last two also pin the item's hazard rule: widening is
    applied to the RESOLVED set after `expand`, so a stale or poisoned row
    drops silently and the turn still compiles — a job that will not boot is
    the failure `capabilities._validate_declarations` chose import-time for,
    not a per-run one.
    """
    store = _fixture_grant_store(tmp_path)
    for scope, pattern in ((OTHER_SCOPE, GRANT_NAME),        # another job's row
                           (THIS_SCOPE, "automod_start"),    # standing worker ban
                           (THIS_SCOPE, "not_a_real_tool")):  # no server serves it
        store.mint(scope=scope, tool_pattern=pattern, issued_by="test-#2471",
                   expires_at=_FAR_FUTURE, note="clause-5 fixture")
    _use_store(monkeypatch, store)
    _bind_scope(request, THIS_SCOPE)

    opts = _sched_opts()   # compiles — does not raise — with all three rows live
    assert opts.allowed_tools is not None
    for smuggled in (GRANT_NAME, "automod_start", "not_a_real_tool"):
        assert smuggled not in opts.allowed_tools, (
            f"{smuggled!r} reached this turn's advertised set from a row that "
            "does not license it")
    assert "Bash" in opts.allowed_tools, "the negatives took the base set with them"


# ══════════════════ #2670: the ceiling reaches the live scheduled-task turn ══════════════════
#
# Everything above is the ceiling as the WORKER POOL compiles it. What follows is
# the ceiling reaching the one turn that had never been under it:
# `app/autonomy.py::run_task`, which predates this machinery and assembled its own
# `RunOptions` off a deny floor (`_get_disallowed_tools()` + the grant-mint ban).
# `scheduled-task` therefore had a declared 33-name ceiling and no denominator: no
# turn was ever compiled against it, so the seven-day check #2471 owes could not
# be computed at all. #2670 wires the turn to `envelope_for` in SHADOW mode — the
# ceiling is compiled and every name it would have refused is recorded to a durable
# ledger, while the turn is still dispatched under exactly the set it was
# dispatched under before this commit.
#
# Process boundaries this section crosses, each with a node below:
#   run_task → `RunOptions` (in-process, and the seam the clause is about): the
#     dispatched set is asserted through the real `_pre_dispatch`, never a spy.
#   `capabilities.envelope_for` → run_task: asserted with a counting proxy that
#     delegates to the real function, because "the ceiling is what reaches the
#     options" is a claim about ONE SPECIFIC CALL and its arguments, not about a
#     value that could have been assembled anywhere.
#   run_task → the ceiling ledger: a file under the data root that a DIFFERENT
#     process — the owed-check job, days later — reads to decide whether
#     enforcement may start. Asserted as bytes at a path, in the shape that reader
#     parses, not as an in-memory call.
SCHED_SOURCE = "scheduled-task"
SCHED_TASK_ID = 77

#: A tier-2 durable-external name the declared ceiling does NOT contain: the name
#: a nightly job's model reaches for when it wants to mail a report, and the row
#: in the ledger that says so. NOT usable as a dispatch probe — `run_task` arms the
#: policy hook with the task's grant scope, so an ungranted tier-2 call is refused
#: by the GRANT gate on a scheduled turn whatever the ceiling says, and this file's
#: own rule is that a refusal by an unrelated rail is not a measurement of the
#: envelope. Asserted, not assumed, at the bottom of
#: `test_a_shadow_turn_is_dispatched_under_exactly_todays_deny_floor`.
OUTSIDE_CEILING = "email_send"

#: The dispatch probe: a real served tool the 33-name ceiling excludes that no
#: other rail on this turn objects to (tier-1, not a standing ban, not grant-
#: gated), so a `None` from `_pre_dispatch` below means the ENVELOPE allowed it and
#: a refusal names the envelope. `browser_navigate` is the desktop/browser family,
#: which no scheduled task has ever had a reason to hold.
DISPATCH_PROBE = "browser_navigate"


def _sched_task(**over) -> dict:
    task = {"id": SCHED_TASK_ID, "name": "Nightly vault landing",
            "skill_name": "nightly-vault-maintenance", "status": "up_next",
            "timeout_seconds": 300}
    task.update(over)
    return task


@pytest.fixture()
def sched_turn(tmp_path, monkeypatch):
    """One real `run_task` turn, with everything it would write OUTSIDE the
    ceiling isolated and the ceiling itself left completely real.

    The stubs are the run's side effects on the fleet (task frontmatter, run
    records, sessions, event log) — nothing that a clause below reads. The
    ledger and the grant store are pointed at `tmp_path` rather than stubbed:
    the row's whole job is to outlive this process, so it is written and read
    back as bytes, and the grant rows are what the widening reads, so they are
    real rows in a real sqlite file (`LLOYD_GRANT_DB` is the knob every other
    grant suite uses; unset, `default_store()` resolves to the production
    `workers.db`, which a test must never open for writing).
    """
    (tmp_path / "config.yaml").write_text(
        "agent:\n  max_turns: 60\nmodel:\n  default: primary\n")
    from app import autonomy
    monkeypatch.setattr(autonomy, "LLOYD_HOME", tmp_path)
    for name in ("_update_task_field", "_append_activity_log"):
        monkeypatch.setattr(autonomy, name, lambda *a, **k: None)
    # `_record_phrase`'s return is concatenated into the failure note, so this
    # stub answers with the empty string rather than None.
    monkeypatch.setattr(autonomy, "_record_phrase", lambda *a, **k: "")
    monkeypatch.setattr(autonomy, "_write_run_record",
                        lambda **kw: {"status": "recorded"})
    monkeypatch.setattr(autonomy, "_get_model_env",
                        lambda *a, **k: {"ANTHROPIC_BASE_URL":
                                        "http://127.0.0.1:8096"})
    monkeypatch.setattr(autonomy, "_load_skill_content",
                        lambda sn: "# fixture skill body")
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    from app.harness.policy import GrantStore
    grants = GrantStore(tmp_path / "grants.db")
    monkeypatch.setenv("LLOYD_GRANT_DB", str(grants.db_path))
    ledger = tmp_path / "scheduling" / "scheduled-ceiling.jsonl"
    from app import scheduled_ceiling as sc
    monkeypatch.setenv(sc.LEDGER_ENV, str(ledger))
    monkeypatch.delenv(sc.MODE_ENV, raising=False)
    return {"root": tmp_path, "grants": grants, "ledger": ledger}


def _drive_sched_turn(monkeypatch, task, *, calls, mode=None, spy=None):
    """Run one `run_task` turn whose model calls `calls`, and hand back its options.

    The harness is the only stub: `run_query` yields the events a turn that made
    those calls would have streamed. Everything between that stub and the ledger
    file is production code, including the ceiling compile, the options and the
    row write.
    """
    import asyncio
    import app.harness as harness_mod
    from app import autonomy
    from app import scheduled_ceiling as sc

    if mode is not None:
        monkeypatch.setenv(sc.MODE_ENV, mode)
    events: list[dict] = []
    for i, name in enumerate(calls):
        events.append({"type": "tool_call", "call_id": f"c{i}", "name": name})
        events.append({"type": "tool_result", "call_id": f"c{i}", "name": name,
                       "content": "ok"})
    events.append({"type": "text_delta", "text": "done"})
    events.append({"type": "result", "stop_reason": "stop", "num_turns": 1,
                   "usage": {}})
    captured: dict = {}

    async def _run_query(messages, options):
        captured["options"] = options
        for evt in events:
            yield evt

    monkeypatch.setattr(harness_mod, "run_query", _run_query)
    monkeypatch.setattr(autonomy, "_find_task_file",
                        lambda tid: Path("task.md"))
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: dict(task))
    if spy is not None:
        real = caps.envelope_for

        def _proxy(source, extra_disallowed=(), *, grant_widening=(), scope=""):
            result = real(source, extra_disallowed,
                          grant_widening=grant_widening, scope=scope)
            spy.append({"source": source,
                        "extra_disallowed": list(extra_disallowed or ()),
                        "grant_widening": tuple(grant_widening), "scope": scope,
                        "result": result})
            return result

        monkeypatch.setattr(caps, "envelope_for", _proxy)
    result = asyncio.run(autonomy.run_task(task["id"]))
    assert result.get("success"), result
    return captured["options"]


def test_run_task_compiles_its_tool_set_through_envelope_for(sched_turn,
                                                             monkeypatch):
    """Clause 1: the `envelope_for` call is what the turn is built from.

    Asserted about the call, not about a number: same source, same grant scope as
    the policy hook was armed with, widening read as the live rows for exactly
    this task's scope. A locally-assembled list of the right length would pass a
    value assertion and fail this one.
    """
    from app.harness.capabilities import live_grant_names
    spy: list[dict] = []
    scope = f"autonomy-task:{SCHED_TASK_ID}"
    sched_turn["grants"].mint(
        scope=scope, tool_pattern=OUTSIDE_CEILING, issued_by="Alan",
        expires_at="2099-01-01T00:00:00+00:00", note="fixture")
    opts = _drive_sched_turn(monkeypatch, _sched_task(), calls=["Bash"],
                             spy=spy)
    assert len(spy) == 1, f"expected exactly one ceiling compile, got {spy}"
    call = spy[0]
    assert call["source"] == SCHED_SOURCE
    assert call["scope"] == scope == opts.grant_scope, (
        "the ceiling was compiled for a different identity than the grant gate "
        "was armed with — the envelope and the gate would be answering for "
        "different jobs")
    assert call["grant_widening"] == live_grant_names(sched_turn["grants"],
                                                     scope)
    assert OUTSIDE_CEILING in call["grant_widening"], (
        "a live grant row for this task did not reach the widening, so the "
        "envelope would be the base set and the job's own authority invisible")
    assert call["extra_disallowed"], (
        "the ceiling must be narrowed by the deny floor the same turn carries, "
        "not compiled beside it")
    assert opts.allowed_tools is None, (
        "clause 2: the shipped default is shadow, and shadow must not dispatch "
        "an allow-list")
    assert len(spy[0]["result"]) == 33 + 1, (
        "the declared ceiling is 33 names; this turn's envelope is that plus the "
        "one live grant row, which is the widening working as designed")


def test_the_compiled_envelope_is_what_reaches_run_options_under_enforce(
        sched_turn, monkeypatch):
    """Clause 1's other edge: when enforcement is on, that same call's result IS
    `RunOptions.allowed_tools` — the wiring is not shadow-only decoration.

    #2670's owed entry 1 asks the later enforcing commit to assert
    `allowed_tools == envelope_for(...)`. It is pinned here as well, because the
    line that would break is this one, and a wiring that only works in the mode
    nobody dispatches under is the failure mode this file exists to catch.
    """
    spy: list[dict] = []
    opts = _drive_sched_turn(monkeypatch, _sched_task(), calls=["Bash"],
                             mode="enforce", spy=spy)
    assert len(spy) == 1, spy
    assert opts.allowed_tools == spy[0]["result"]
    assert opts.allowed_tools == caps.envelope_for(
        SCHED_SOURCE, opts.disallowed_tools,
        scope=f"autonomy-task:{SCHED_TASK_ID}")


def test_a_shadow_turn_is_dispatched_under_exactly_todays_deny_floor(
        sched_turn, monkeypatch):
    """Clause 2: no scheduled task loses a tool in this commit.

    Not asserted as `allowed_tools is None` alone — that would be true of a
    compile that never happened. The dispatched set is computed the way the loop
    computes it (`_reach_after`) and compared to the deny floor's reach
    (`_reach_before`), and a tier-2 tool the ceiling excludes is put through the
    real `_pre_dispatch`: in shadow it goes through to the MCP layer, which is
    what "unchanged from today" means operationally.
    """
    assert OUTSIDE_CEILING not in (caps.envelope_for(SCHED_SOURCE) or []), (
        "the control below is only a control if the ceiling excludes the name")
    opts = _drive_sched_turn(monkeypatch, _sched_task(),
                             calls=["Bash", OUTSIDE_CEILING])
    assert _reach_after(opts) == _reach_before(opts)
    assert _allowed_names(opts) is None
    assert DISPATCH_PROBE not in (caps.envelope_for(SCHED_SOURCE) or [])
    assert _dispatch(opts, DISPATCH_PROBE) is None, (
        "a shadow turn must still dispatch a name the ceiling would strip — "
        "refusing it here is the regression clause 2 forbids")
    # The rail this probe deliberately avoids, named so the probe above cannot be
    # read as "scheduled tasks can send email": the grant gate already refuses an
    # ungranted tier-2 call on a scheduled turn, ceiling or no ceiling.
    grant_refusal = _dispatch(opts, OUTSIDE_CEILING)
    assert grant_refusal is not None and "grant" in str(grant_refusal)


def test_the_same_turn_under_enforce_proves_the_shadow_assertion_is_not_vacuous(
        sched_turn, monkeypatch):
    """The control for the node above: one env var, and the same tool is refused.

    Without this the equality above could be describing a ceiling that was never
    compiled. Under `enforce` the dispatched set is the envelope intersected with
    the floor, strictly smaller than the floor's reach, and `_pre_dispatch`
    refuses `email_send` with the envelope's own message.
    """
    opts = _drive_sched_turn(monkeypatch, _sched_task(), calls=["Bash"],
                             mode="enforce")
    reach = _reach_after(opts)
    assert reach < _reach_before(opts)
    assert reach == _reach_before(opts) & _allowed_names(opts)
    refusal = _dispatch(opts, DISPATCH_PROBE)
    assert refusal is not None and "not available on this turn" in str(refusal)


def _sched_rows(sched_turn) -> list[dict]:
    from app import scheduled_ceiling as sc
    return sc.read_rows(sched_turn["ledger"])


def test_a_shadow_turn_records_the_names_the_ceiling_would_strip(sched_turn,
                                                                monkeypatch):
    """Clause 3: one durable row per shadow turn, naming what fell outside.

    This is the artifact enforcement is gated on, and shadow is precisely the
    mode that leaves no trace in the transcript: nothing refused the call, so
    the session file of a shadow turn is byte-identical to one with no ceiling.
    A row per turn, with the date, the task id, the scope, the mode and the
    out-of-envelope names.
    """
    import datetime
    opts = _drive_sched_turn(monkeypatch, _sched_task(),
                             calls=["Bash", "Read", OUTSIDE_CEILING])
    assert opts.allowed_tools is None
    rows = _sched_rows(sched_turn)
    assert len(rows) == 1, f"exactly one row per turn, got {rows}"
    row = rows[0]
    assert row["task_id"] == SCHED_TASK_ID
    assert row["grant_scope"] == f"autonomy-task:{SCHED_TASK_ID}"
    assert row["mode"] == "shadow"
    assert row["date"] == datetime.datetime.now(
        datetime.timezone.utc).date().isoformat()
    assert row["would_strip"] == [OUTSIDE_CEILING]
    assert row["n_would_strip"] == 1
    assert row["envelope_size"] == 33


def test_a_turn_whose_calls_are_all_inside_the_ceiling_writes_a_row_with_no_names(
        sched_turn, monkeypatch):
    """Clause 3's other half, and the one the 5% rate depends on.

    A clean turn must be a ROW with an empty name list, not the absence of a
    row: the flip gate divides would-strip occurrences by turns, and writing
    only the interesting turns would silently shrink the denominator to the set
    of failures, which is how a rate gets published as 100% of nothing.
    """
    _drive_sched_turn(monkeypatch, _sched_task(), calls=["Bash", "Read"])
    rows = _sched_rows(sched_turn)
    assert len(rows) == 1, rows
    assert rows[0]["would_strip"] == []
    assert rows[0]["n_would_strip"] == 0
    assert "compile_error" not in rows[0]


def test_every_turn_writes_exactly_one_row_including_the_ones_that_crash(
        sched_turn, monkeypatch):
    """Clause 3's "exactly one": two clean turns are two rows, and a run that
    raises mid-turn still writes its row.

    The row is written in `run_task`'s `finally`, and the exit paths there are
    six (success, timeout, cancel, empty response, infra failure, exception). A
    row written on the success path only would drop exactly the runs whose tool
    use most needs measuring, and the rate would be computed over the traffic
    that behaved.
    """
    import asyncio
    import app.harness as harness_mod
    from app import autonomy

    _drive_sched_turn(monkeypatch, _sched_task(), calls=["Bash"])
    _drive_sched_turn(monkeypatch, _sched_task(), calls=["Bash", "Read"])
    rows = _sched_rows(sched_turn)
    assert len(rows) == 2, rows
    assert [r["n_would_strip"] for r in rows] == [0, 0]

    async def _run_query_boom(messages, options):
        yield {"type": "tool_call", "call_id": "c0", "name": OUTSIDE_CEILING}
        raise RuntimeError("engine died mid-turn")

    monkeypatch.setattr(harness_mod, "run_query", _run_query_boom)
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: Path("task.md"))
    monkeypatch.setattr(autonomy, "_parse_task_file",
                        lambda p: _sched_task(failure_count=0, max_retries=9))
    asyncio.run(autonomy.run_task(SCHED_TASK_ID))
    rows = _sched_rows(sched_turn)
    assert len(rows) == 3, f"the crashed turn owes a row too, got {rows}"
    assert rows[-1]["would_strip"] == [OUTSIDE_CEILING]
    # A run id, not a unique one: `run_77_<second>` collides when three turns run
    # inside one second, which only a test does.
    assert rows[-1]["run_id"] and rows[-1]["mode"] == "shadow"


def test_a_task_that_declares_the_vault_land_grant_is_never_counted_as_would_strip_it(
        sched_turn, monkeypatch):
    """Clause 4, end to end: the nightly vault-landing job is not a would-strip.

    The whole reason this exception exists. The task file declares the grant,
    `sync_task_grants` materializes the row, the widening reads it back for this
    scope, and a turn that calls `automod_vault_land` is recorded as clean. Strip
    the exception and this is the job the ceiling would have broken at flip time —
    the regression `capabilities.py` warned about in prose for four days.
    """
    task = _sched_task(grants=[{"tool": "automod_vault_land",
                                "expires_at": "2099-01-01T00:00:00+00:00",
                                "issued_by": "Alan", "note": "vault lander"}])
    _drive_sched_turn(monkeypatch, task,
                      calls=["Bash", "vault_write", "automod_vault_land"])
    rows = _sched_rows(sched_turn)
    assert len(rows) == 1, rows
    assert rows[0]["would_strip"] == [], (
        "a job whose own frontmatter declares the tool must not be counted as "
        "reaching outside its ceiling")
    assert rows[0]["envelope_size"] == 34, (
        "33 declared names plus the granted exception, and nothing else")


def test_an_autonomy_task_scope_with_a_live_row_compiles_the_exception_in(
        sched_turn):
    """Clause 4, at the compile: the exception is reachable, but only through a
    live row for THAT scope — never through the source, and never across jobs.

    `expand` still refuses a standing ban in a declared set (the anti-smuggling
    rule #2471 chose deliberately), so a widening is the only route, and
    `_widening_names` drops standing bans unless the scope prefix is in
    `SCOPE_BAN_EXCEPTIONS`. Both halves are asserted here: the row admits the
    tool, and its absence — or another job's row — does not.
    """
    from app.harness.capabilities import live_grant_names
    scope = f"autonomy-task:{SCHED_TASK_ID}"
    base = caps.envelope_for(SCHED_SOURCE)
    assert "automod_vault_land" not in base
    # The class from the module object, not the one this file imported at load
    # time: another suite reloads `app.harness.capabilities`, and a reloaded
    # module's `UnknownCapability` is a different object from the stale binding —
    # which shows up as this assertion "failing" on an exception it did raise.
    with pytest.raises(caps.UnknownCapability):
        caps.expand(("automod_vault_land",))
    sched_turn["grants"].mint(
        scope=scope, tool_pattern="automod_vault_land", issued_by="Alan",
        expires_at="2099-01-01T00:00:00+00:00", note="fixture")
    widening = live_grant_names(sched_turn["grants"], scope)
    assert widening == ("automod_vault_land",)
    env = caps.envelope_for(SCHED_SOURCE, grant_widening=widening, scope=scope)
    assert "automod_vault_land" in env
    assert len(env) == len(base) + 1
    # The same row, read for a job that does not hold it, widens nothing.
    assert live_grant_names(sched_turn["grants"], "autonomy-task:9999") == ()
    assert "automod_vault_land" not in caps.envelope_for(
        SCHED_SOURCE, grant_widening=(), scope="autonomy-task:9999")
    # An expired row is not a licence: `live_grant_names` is the reader, and it
    # filters on expiry, so the exception cannot outlive the grant that earned it.
    sched_turn["grants"].mint(
        scope="autonomy-task:99", tool_pattern="automod_vault_land",
        issued_by="Alan", expires_at="2020-01-01T00:00:00+00:00",
        note="fixture, expired")
    assert live_grant_names(sched_turn["grants"], "autonomy-task:99") == ()


def test_no_worker_source_envelope_names_the_vault_land_exception():
    """Clause 4's other half: the exception is a per-job crack, not a hole.

    Every declared source's envelope is compiled the way the worker pool compiles
    it — unscoped — and none of them may name the tool. This is also the node that
    keeps the baseline artifact honest: `scheduled-task`'s row there is built by
    the same unscoped call, so the exception cannot have quietly widened the
    fleet's published numbers.
    """
    for source in sorted(caps.SOURCE_CAPABILITIES):
        env = caps.envelope_for(source)
        assert env is not None and "automod_vault_land" not in env, source
    for source in sorted(caps.SOURCE_CAPABILITIES):
        env = caps.envelope_for(SCHED_SOURCE,
                                grant_widening=("automod_vault_land",),
                                scope=f"worker:{source}")
        assert "automod_vault_land" not in env, source
    assert caps.scoped_ban_exceptions("worker:scheduled-task") == frozenset()
    assert caps.scoped_ban_exceptions("") == frozenset()
    assert caps.scoped_ban_exceptions("autonomy-task:7") == frozenset(
        {"automod_vault_land"})


def test_the_default_mode_is_shadow_and_an_unknown_value_stays_shadow(
        monkeypatch):
    """The switch fails toward "record only", in both directions of a mistake.

    Unset is shadow, `SHADOW` is shadow (case-insensitive by intent), and a typo
    is shadow rather than enforcement — a mis-typed env var must not be the act
    that starts refusing a fleet's tools. The mode is written into every row, so
    an operator who does set `enforce` by hand is still visible in the evidence
    the flip gate reads.
    """
    from app import scheduled_ceiling as sc
    monkeypatch.delenv(sc.MODE_ENV, raising=False)
    assert sc.mode() == "shadow"
    monkeypatch.setenv(sc.MODE_ENV, "SHADOW")
    assert sc.mode() == "shadow"
    monkeypatch.setenv(sc.MODE_ENV, "enforce-now")
    assert sc.mode() == "shadow"
    monkeypatch.setenv(sc.MODE_ENV, "enforce")
    assert sc.mode() == "enforce"


# ════════════════════════════════════════════════════════════════════════════
#  #2671: autotriage gets a declared ceiling.
#
#  The denominator, recorded once here so every number below is re-checkable:
#  every `*_autotriage_*.json` and `*_autotriage_*.json.gz` under
#  `~/lloyd-data/sessions` on 2026-10-14 — 1734 files, all parsed, 32,814 tool
#  calls, counted as distinct FILES containing a name (the #2471 denominator rule).
#  The >5%-of-files line is 87 files. Above it: Bash 1714, backlog_write_task
#  1176, Read 455, Grep 191, backlog_get_task 180. Its volume is why this source
#  went first among the undeclared twelve — 1615 runs in `workers.db` since
#  2026-10-06, against 622 for owed-check and 438 for autocode.
# ════════════════════════════════════════════════════════════════════════════
#: The five names the census puts above the 87-file line. This is the floor the
#: declaration has to clear, so it is asserted rather than narrated: a later
#: round that narrowed the base past one of these turns the triage loop off at
#: the tool it cannot work without, and this is the node that says so.
TRIAGE_CENSUS_ABOVE_LINE = ("Bash", "backlog_write_task", "Read", "Grep",
                            "backlog_get_task")

#: The four names clause 2 names. Chosen because none is in autotriage's declared
#: base and each is durable-outside-the-box: mail, contacts, calendar, and the
#: autonomy mutator whose file the fleet itself schedules.
TRIAGE_REFUSED_BY_NAME = ("email_send", "contacts_delete",
                          "calendar_delete_event", "autonomy_write_task")


def _triage_opts():
    """`autotriage`'s real single-item options, through the one compiler."""
    return C._worker_run_options(4, source=TRIAGE_SOURCE)


def _triage_sweep_opts():
    """The same base narrowed the way the sweep path narrows it."""
    return C._worker_run_options(
        4, source=TRIAGE_SOURCE, extra_disallowed=list(autotriage.SWEEP_DISALLOWED))


def test_autotriage_gets_a_declared_ceiling_its_turn_options_carry():
    """Clause 1: `envelope_for` stops returning None, and the options carry it.

    Before #2671 this source reached 133 of the 147-name served pool — every
    name the deny union forgot to name, including 26 that write outside the box.
    The ceiling is the census: the five above-line names, plus the below-line
    names that are the job's own surface (`Write` 62 files / 104 calls, 100 of
    them a `/tmp/*.py` verification script the prompt's own ceiling bullet
    instructs and the other 4 a scratch probe elsewhere; `backlog_tasks` 27;
    `Scratchpad` 24; `automod_status` 21, which the fleet bans the loop DRIVERS of
    but not this read-only status tool; `Edit` 13; `skills_search` 6;
    `vault_read` 6; `skills_read` 4; `vault_search` 3; `memory_read` 3), expressed
    as the existing families where a family is what it means — `READ_FILES`,
    `READ_VAULT`, `FILE_BACKLOG`, `TURN_STATE`, `READ_MEMORY`.

    What is deliberately NOT here, with its census count, is in the entry in
    `SOURCE_CAPABILITIES` beside the tuple; the owed-check ruling on whether the
    base was wide enough reads that record.
    """
    env = caps.envelope_for(TRIAGE_SOURCE)
    assert isinstance(env, list) and env, \
        "autotriage is still compiled from the deny list alone"
    assert env == sorted(set(env))
    assert env, "envelope_for returned an empty list"
    assert all(n in POOL for n in env), sorted(n for n in env if n not in POOL)
    for name in TRIAGE_CENSUS_ABOVE_LINE:
        assert name in env, (
            f"{name} is above the census's 87-file line and left the base")
    # Read-only over Lloyd's own stores, below the line but plainly the job.
    for name in ("Write", "Edit", "backlog_tasks", "Scratchpad", "automod_status",
                 "Glob", "skills_search", "skills_read", "vault_read",
                 "vault_search", "memory_read", "backlog_get_task",
                 "backlog_write_task", "Read", "Grep", "Bash"):
        assert name in env, name
    assert "vault_write" not in env, \
        "the vault route is automation-only; autotriage called it 0/1734 times"

    opts = _triage_opts()
    assert opts.allowed_tools == env, opts.allowed_tools
    # The compile actually narrows: nothing the standing bans deny is in the
    # base, and the reach is the base, not the pool.
    denied = {normalize_tool_name(d) for d in (opts.disallowed_tools or [])}
    assert not (denied & set(env)), sorted(denied & set(env))
    assert _reach_after(opts) == set(env), (
        len(_reach_before(opts)), len(_reach_after(opts)))


@pytest.mark.parametrize("tool", TRIAGE_REFUSED_BY_NAME)
def test_autotriage_base_refuses_every_durable_external_name_by_name(tool):
    """Clause 2: the ceiling reaches no tier-2/3 durable-external name, by name.

    This is the only test node that pins it for `autotriage` (#2624 was the
    fleet-wide one, and it was green with this source undeclared because the
    matrix exempts an undeclared source from the durable-external row — there is
    no base to grade). The refusal is asserted as the envelope's own text rather
    than any refusal, because three of these four are ALSO refused on an
    unattended turn by the grant gate, and a node that accepted "denied" from
    either rail would have been green before the declaration existed — the
    vacuous-pass trap #2671's review rung exists to catch. The paired node below
    is the control that the message is not a coincidence.
    """
    declared = caps.SOURCE_CAPABILITIES[TRIAGE_SOURCE]
    reach = caps.expand(declared)
    assert tool not in reach
    assert not caps.durable_external(reach), sorted(caps.durable_external(reach))
    evt = _dispatch(_triage_opts(), tool)
    assert evt is not None, f"{tool} still dispatches on a triage turn"
    assert "not available on this turn" in evt["content"], evt["content"]


def test_the_durable_names_autotriage_lost_are_ones_an_undeclared_source_keeps():
    """The control for the node above: the same refusal on `arch-review`'s options.

    `arch-review` declares nothing, so its turn still reaches all four names.
    Three of them are refused at dispatch by the GRANT gate on any unattended
    turn, which is exactly why the control is asserted as reachability, not as a
    dispatch: on a reachability control "it was refused" would pass on both
    turns and prove nothing. The one name the grant gate does NOT refuse on a
    worker-scoped turn — `autonomy_write_task`, tiered by destination, which a
    bare call does not supply — is asserted as a dispatch control as well: it
    reaches the MCP layer on arch-review's turn and is refused by the envelope on
    a triage turn, with the envelope's own message.
    """
    arch = C._worker_run_options(4, source="arch-review")
    assert arch.allowed_tools is None, \
        "the control turned declared; this node is no longer a control"
    for tool in TRIAGE_REFUSED_BY_NAME:
        assert tool in _reach_after(arch), (
            f"{tool} is unreachable on an undeclared turn, so the triage "
            "envelope refused nothing that was not already gone")
        assert tool not in _reach_after(_triage_opts()), tool

    evt_declared = _dispatch(_triage_opts(), "autonomy_write_task")
    assert evt_declared is not None
    assert "not available on this turn" in evt_declared["content"], \
        evt_declared["content"]
    assert _dispatch(arch, "autonomy_write_task") is None, (
        "autonomy_write_task is no longer grant-refused-but-reachable on an "
        "unattended turn; the dispatch control has to be re-picked")
    for tool in ("email_send", "contacts_delete", "calendar_delete_event"):
        evt = _dispatch(arch, tool)
        assert evt is not None and "grant" in evt["content"], (
            f"{tool} on an undeclared turn was refused by something other than "
            "the grant gate: "
            f"{(evt or {}).get('content', 'dispatched')}")


def test_the_baseline_row_for_autotriage_reads_declared_with_no_durable_external():
    """Clause 4: the committed artifact describes the NEW triage compile.

    #2471's round pinned this invariant for the fleet (the artifact must not be a
    record of an older code state); this is the same claim pinned for the one row
    this round moved. `n_allowed == reachable_after` is what "the ceiling is what
    binds" means in the artifact's own terms, and 26 durable-external names
    dropping to 0 is the exposure #2671 existed to close.
    """
    art = _table()
    row = art["sources"][TRIAGE_SOURCE]
    base = art["sources"]["arch-review"]      # an undeclared row, for contrast
    assert base["declared"] is False, "an undeclared control row turned declared"
    assert row["declared"] is True, (
        "autotriage is still undeclared in the artifact — re-record: "
        "LLOYD_RECORD_CAPABILITY_BASELINE=1 pytest tests/"
        "test_worker_capability_allowlist.py")
    assert row["n_allowed"] == row["reachable_after"], row
    assert row["durable_external_after"] == [], row["durable_external_after"]
    assert row["reachable_after"] < row["reachable_before"], row
    assert set(row["durable_external_before"]) - set(row["durable_external_after"]), row


def test_the_sweep_narrowing_reaches_the_post_rather_than_widening_the_base(
        monkeypatch):
    """The read-only sweep keeps its narrowing across the one real seam.

    `envelope_for` subtracts `extra_disallowed` AFTER expanding the base, so
    declaring `FILE_BACKLOG` writes for the single-item and group paths could in
    principle have re-widened the sweep — the exact leak #2671 had to be checked
    against. It does not: the body the shipped `run_prompt_in_session` builds for
    the sweep's arguments carries the narrowed list, and the three writes the
    sweep forbids are absent from the ceiling the turn is advertised as well as
    from its deny floor — the leak would have been a ceiling that named them
    while the deny floor refused them, because the turn is *told* the ceiling
    (#1066). On the sweep turn `backlog_write_task`, `Write` and `Edit` are
    refused by the deny floor's own message, as they have always been, while
    `Read`, `Grep` and `Bash` still dispatch. The sweep prompt says
    `SWEEP_DISALLOWED` makes the turn read-only "at every turn shape", which this
    node is the evidence for.
    """
    posted = _post_body(TRIAGE_SOURCE, list(autotriage.SWEEP_DISALLOWED),
                        monkeypatch)
    assert posted["platform"] == "worker", posted
    assert posted["allowed_tools"] == caps.envelope_for(
        TRIAGE_SOURCE, list(autotriage.SWEEP_DISALLOWED)), posted["allowed_tools"]
    assert set(posted["allowed_tools"]) < set(
        caps.envelope_for(TRIAGE_SOURCE)), posted["allowed_tools"]
    for name in ("Write", "Edit", "backlog_write_task", "vault_write",
                 "autonomy_write_task"):
        assert name not in posted["allowed_tools"], name
    # And the turn shape with no narrowing keeps the writes the sweep forbids —
    # so the ceiling is one base narrowed per turn shape, not a second ceiling
    # nobody declared.
    plain = _post_body(TRIAGE_SOURCE, [], monkeypatch)
    assert plain["allowed_tools"] == caps.envelope_for(TRIAGE_SOURCE), \
        plain["allowed_tools"]
    for name in ("Write", "Edit", "backlog_write_task"):
        assert name in plain["allowed_tools"], name

    opts = _triage_sweep_opts()
    for name in ("Write", "Edit", "backlog_write_task"):
        evt = _dispatch(opts, name)
        assert evt is not None and "disabled by configuration" in evt["content"], \
            (name, (evt or {}).get("content", "dispatched"))
    for name, args in (("Read", None), ("Bash", {"command": "ls"}),
                       ("Grep", None)):
        assert _dispatch(opts, name, args=args) is None, name


def test_autotriage_is_declared_without_joining_the_untrusted_fleet_table():
    """The scope line #2671 drew: one source, and not onto the protected list.

    `autotriage` reads Lloyd's own backlog, not the internet, so it is not an
    untrusted-ingest source; the item's sixth clause is "one source only this
    round", and widening the fleet's most-protected roster is a different
    round's ruling. What IS asserted is the direction the ceiling moves for the
    sources that were already declared: their compiles are untouched, because
    every name in the new base is either a family or a name no ban reaches.
    """
    assert TRIAGE_SOURCE not in UNTRUSTED
    assert TRIAGE_SOURCE not in caps.UNTRUSTED_INGEST_SOURCES
    for src in UNTRUSTED:
        env = caps.envelope_for(src)
        assert env is None or set(env) <= POOL, src


# ── #2718: bench-mine — the source whose ceiling IS the fix ──────────────────
#
# Every other declaration in this file narrowed a surface that happened to be wide.
# This one closes a hole that was open *because* the surface was wide. On 2026-10-13
# session `20261013_225121_benchmine_2a67` (a `bench_027` loser mine) tried to `Write` a
# bench draft, the bench-assertion rail refused it and told it to add the missing key to
# `eval/autoresearch_assertions.yaml` and retry, and the turn obeyed: a `python3` heredoc
# through `Bash` appended 8 lines to that tracked file of the live checkout (msg 46's own
# reasoning: "the rail explicitly told me to add the key and retry — that is the way
# forward"). The source declared nothing, so `envelope_for` answered None and the turn
# shipped the floor-only compile, which contains `Write`, `Edit` and `Bash`. Its contract
# is text-to-staging (`_TASK_SCHEMA`: "Output ONLY the full markdown file content"), so
# the ceiling below is not a judgement about how much a miner needs; it is the contract.
#
# Deliberately NOT done, and named so the next reader does not "finish" it:
# `agent_mcp/_tool_sandbox.py` still keeps `benchmine` out of
# `SANDBOXED_BACKGROUND_SLUGS`, which is a fleet-wide read-only hammer that would also
# refuse every non-`readOnlyHint` MCP tool and `Scratchpad`. The item's triage measured
# that blast radius as larger than the hole.

_REPO = Path(__file__).resolve().parents[1]
BENCH_NAME = "bench-mine"
BENCH_ENVELOPE = ["Glob", "Grep", "Read", "Scratchpad", "recall_observation"]
#: Anything here that survived into the compile is a way back into a file. `Bash` is on
#: the list because the incident's write was a shell heredoc, not because a shell is a
#: file-mutating tool by name.
BENCH_FORBIDDEN = {"Write", "Edit", "Bash", "NotebookEdit", "apply_patch"}


def test_bench_mine_declares_a_write_free_ceiling_and_no_longer_returns_none():
    """#2718 clause 1: the lookup answers a ceiling, and the ceiling has no write path.

    Three things, because `is not None` alone is the weak half of this clause — an
    envelope that declared `Write` would satisfy it. (a) The lookup answers, so the
    miner's compile is the ceiling and not the floor. (b) No write-capable name is
    reachable: the five names are `READ_FILES` (Read/Grep/Glob) plus `TURN_STATE`
    (Scratchpad, recall_observation), nothing else — and `ToolSearch` is among the names
    the ceiling does NOT grant, which is the point of a compile this small: a miner turn
    has no route to load an MCP tool it was not given. (c) None of those five reaches a durable
    external write through the registry — the matrix computes that independently, so a
    name that gained a mutating capability later is caught here rather than in this
    comment.
    """
    env = caps.envelope_for(BENCH_NAME)
    assert env is not None, (
        "bench-mine declares no capability set: an undeclared source compiles the "
        "floor-only surface, which is where Write and Bash came from")
    assert env == BENCH_ENVELOPE, env
    assert not (set(env) & BENCH_FORBIDDEN), env
    assert set(caps.SOURCE_CAPABILITIES[BENCH_NAME]) == {"READ_FILES", "TURN_STATE"}, \
        caps.SOURCE_CAPABILITIES[BENCH_NAME]
    durable = {n for n in set(env) & (set(TIER2_TOOLS) | set(TIER3_TOOLS))}
    assert durable == set(), (
        f"a durable-external name is in the miner's ceiling: {sorted(durable)}")
    # The ruling this round did NOT take: a per-source ceiling, not a fleet-wide one. A
    # sibling worker source is still undeclared, so `bench-mine` was not swept in with a
    # wider change, and the queue order for the rest is still owed (#2269 owed entry 2).
    assert "arch-review" not in caps.SOURCE_CAPABILITIES


def test_the_bench_mine_ceiling_leaves_the_miner_its_reads_and_its_scratchpad():
    """The other edge: the ceiling is narrow, and it is not narrower than the job.

    The prompt the source ships says "Read that file with the Read tool"
    (`workers/sources/bench_mine.py`, the loser-mine prompt), so `Read` has to survive or
    the ceiling breaks the job — and `Scratchpad` has to survive because it is not in
    `agent_mcp/annotations.py`'s `READ_ONLY` set, which is exactly why the item's triage
    ruled out routing this through the read-only sandbox instead.
    """
    env = set(caps.envelope_for(BENCH_NAME))
    for name in ("Read", "Grep", "Glob", "Scratchpad", "recall_observation"):
        assert name in env, name
    src = (_REPO / "workers" / "sources" / "bench_mine.py").read_text(encoding="utf-8")
    assert "Read that file with the Read tool" in src, (
        "the miner prompt no longer tells the turn to Read, so the Read in this ceiling "
        "is no longer pinned to a prompt line — re-derive what the turn needs")


def test_the_bench_mine_declare_still_ships_the_automod_ban_and_the_rollover():
    """The one thing the ceiling may not take away, stated as an assertion.

    A capability ceiling is an ALLOW list, and the prompt's claim is that the ban is not
    a ban-shaped surprise inside it: `workers/sources/_common.py` unions
    `WORKER_AUTOMOD_BAN` into `disallowed_tools` on every worker compile regardless of
    source, and `test_scheduled_task_ceiling…` (same file, above) is the same control for
    the first declared source. If the ban were expressed as "absent from the ceiling", a
    later widening could put an automod name back with nothing else noticing; here it is
    not in the ceiling AND the ban is still dispatched-refused, and the tool-rollover
    names in the ban are still refused as names.
    """
    env = set(caps.envelope_for(BENCH_NAME))
    assert not (env & set(C.WORKER_AUTOMOD_BAN)), sorted(env & set(C.WORKER_AUTOMOD_BAN))
    opts = C._worker_run_options(8, source=BENCH_NAME)
    assert opts.allowed_tools == caps.envelope_for(BENCH_NAME)
    assert sorted(set(opts.disallowed_tools) & set(C.WORKER_AUTOMOD_BAN)) == \
        sorted(C.WORKER_AUTOMOD_BAN), opts.disallowed_tools
    for name in list(C.WORKER_AUTOMOD_BAN) + [
            "sessions_spawn", "cron", "gateway", "agents_list", "session_send",
            "session_status", "agents_list"]:
        assert _dispatch(opts, name) is not None, (
            f"{name} dispatched on a miner turn — the ban is no longer reaching this "
            "source, which is the asymmetry the ceiling was supposed to keep")
