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

#: What each source's own job writes. `youtube-digest`'s are the two its prompt
#: names (`Write`/`vault_write` for the note, `fact_add` for the entity);
#: `session-distill`'s is the fact write the distiller exists to make, plus
#: `Bash`, which is how it reads the transcripts it distils (344 of the 560 tool
#: calls in its 54 retained sessions, every one in the last fortnight).
JOBS_MUST_STILL_WORK = {
    "session-distill": ("fact_add", "Bash", "Read"),
    "youtube-digest": ("vault_write", "Write", "fact_add", "backlog_write_task"),
    "deep-research": ("vault_write", "fact_add", "http_fetch"),
}

SOURCE_MODULES = {
    "session-distill": session_distill,
    "youtube-digest": youtube_digest,
    "deep-research": deep_research,
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

@pytest.mark.parametrize("source", UNTRUSTED)
def test_each_source_still_dispatches_the_writes_its_own_job_does(source):
    """The whole failure mode of an allow-list is a job that quietly lost a tool.

    Each name here is one the source's own prompt instructs the turn to call or
    its retained session transcripts show it calling — see the note in
    `app/harness/capabilities.py`. A `None` from `_pre_dispatch` means the call
    reached the MCP layer, which is the only claim this node makes: the grant
    gate, the safety tier of a particular `Bash` command and the content gate
    are other layers with their own suites.
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
    """The JSON body `_stream` would POST, captured at the transport.

    `httpx.AsyncClient` is the socket, not the mechanism under test: what is
    being checked is that the pool's declared envelope is *in the bytes* the
    backend reads. The reply is a single `done` event so the SSE reader exits the
    way it does on a finished turn.
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

    `youtube-digest` and `deep-research` do not build their options in the pool:
    they POST to the backend, which builds them in another process from this
    body and nothing else — the same seam `grant_scope` (#534) and
    `effect_scope` (#544) had to be crossed on their own, for the same reason.
    So a change that stopped at `_worker_run_options` would have shipped an
    envelope for the sources that read a bundle off disk and none for the two
    that read the internet. This node takes the body the real `_stream` builds
    and hands it to the real `build_turn_options`, which is the only thing
    standing between the pool's declaration and the turn that runs.
    """
    import json as _json

    from app.routers import messages as M
    from app.routers import turn_options as topts

    for source in ("youtube-digest", "deep-research"):
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
