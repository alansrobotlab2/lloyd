"""Each worker source's capability envelope, derived from what its job outputs (#2269).

The harness has always been able to run a turn on an allow-list —
`RunOptions.allowed_tools`, folded into the advertised catalog and into every
iteration's dispatch set by `loop._allow_list_hidden`, and refused again in
`_pre_dispatch` for a name no server discovered (P3). Only the memory-flush
turn ever used it. Every worker source instead built a *deny* list: config's
`disabled_tools` + `WORKER_AUTOMOD_BAN` + `WORKER_GRANT_MINT_BAN` + that
source's own `DISALLOWED`, concatenated. A deny list answers "what did someone
think to forbid", so the default for any tool nobody listed is *reachable*:
measured against the live 147-tool pool on 2026-10-06, a `youtube-digest` turn
— whose entire input is a transcript fetched off the internet — reached 99
tools, 12 of them tier-2/3 durable-external (`email_send`, `email_reply`,
`email_forward`, `email_update`, `contacts_create/update/delete`,
`calendar_create/update_event/delete_event`, `tasks_create/update`), none of
which its own RESULT block can use.

This module is the policy that was missing: one declared set per source,
derived from the output that source declares, from which the envelope is
`pool − allowed` rather than `listed`. The enforcement layer stays the single
one it already was; nothing here gates a call.

**Where a declared set comes from.** Not from a guess about what a job "would
want": the three sets below are the union of (a) every tool name the source's
own prompt or RESULT block instructs the turn to call, and (b) every tool name
the retained session transcripts of that source show it actually calling —
counted over `~/lloyd-data/sessions` on 2026-10-06 (101 `youtube-digest`
sessions/982 calls, 45 `deep-research`/1777, 54 `session-distill`/560). A tool
is in a set because the job reached for it or its contract names it; `remember`
is the counter-example that shows the measurement working — 19 `deep-research`
calls named it, no such tool is served today, and a deny list could never have
noticed.

**The residual risk this does not remove, stated plainly.** An untrusted-ingest
turn that legitimately writes durably (a digest writes a note, files a backlog
item) can still be steered into a *bad durable write*. This narrows what an
injected instruction can do; it does not make injection harder to inject —
that stays `run_injection_canary`'s and the outbound content gate's job.
"""

from __future__ import annotations

from app.harness.policy import TIER2_TOOLS, TIER3_TOOLS, normalize_tool_name
from app.harness.tool_roster import REGISTERED_TOOLS

#: What a turn may reach, under a name. A source declares names and set names
#: together; a set is how the same half-dozen observation tools stop being
#: re-typed — and re-typed slightly differently — in every source file.
CAPABILITY_SETS: dict[str, frozenset[str]] = {
    "READ_FILES": frozenset({"Read", "Grep", "Glob"}),
    "READ_VAULT": frozenset({"vault_read", "vault_search", "vault_recall"}),
    "READ_FACTS": frozenset({"fact_get"}),
    "WEB_READ": frozenset({"http_search", "http_fetch"}),
    "BROWSER_READ": frozenset({"browser_navigate", "browser_snapshot",
                               "browser_scroll", "browser_wait", "browser_tabs"}),
    #: The turn's own two read-back affordances: the scratchpad it appends
    #: notes to, and the cleared tool result the relief machinery wrote.
    "TURN_STATE": frozenset({"Scratchpad", "recall_observation"}),
    "READ_MEMORY": frozenset({"memory_read"}),
    "FACT_WRITE": frozenset({"fact_add", "fact_relate", "fact_invalidate"}),
    "FILE_NOTE": frozenset({"Write", "vault_write"}),
    "FILE_BACKLOG": frozenset({"backlog_boards", "backlog_tasks",
                               "backlog_get_task", "backlog_write_task"}),
}


class UnknownCapability(ValueError):
    """A declared name no server serves. Raised at import, never at run time."""


def registered_tool_names() -> frozenset[str]:
    """Every name a worker could be told it may call.

    The committed snapshot unioned with the live discovery universe when a pool
    has been opened in this process: a tool added upstream is therefore usable
    the moment it is served, without waiting for someone to regenerate the
    snapshot, and the snapshot's only possible error is accepting a name that
    then refuses at dispatch — which is loud, where a false negative would be a
    source refusing to boot.
    """
    names = set(REGISTERED_TOOLS)
    try:
        from app.mcp_discovery import _TOOL_UNIVERSE
        names |= {normalize_tool_name(n) for n in _TOOL_UNIVERSE}
    except Exception:  # pragma: no cover - a missing universe is the boot case
        pass
    return frozenset(names)


def expand(declared) -> frozenset[str]:
    """Resolve a declared set (names and set names) to normalised tool names.

    Raises `UnknownCapability` on a name that is neither a set nor a served
    tool, and on a standing worker ban named as a capability — an allow-list
    that "grants" `automod_start` is a bug that must not survive the import,
    even though the deny list compiled beside it would refuse it anyway.
    """
    banned = _standing_bans()
    out: set[str] = set()
    unknown: list[str] = []
    for token in declared or ():
        text = str(token).strip()
        if not text:
            continue
        if text in CAPABILITY_SETS:
            out |= set(CAPABILITY_SETS[text])
            continue
        name = normalize_tool_name(text)
        if name in banned:
            raise UnknownCapability(
                f"{text!r} is a standing worker ban (app.tool_bans); a source "
                "may not declare it as a capability")
        if name not in registered_tool_names():
            unknown.append(text)
            continue
        out.add(name)
    if unknown:
        raise UnknownCapability(
            f"declared capability names no server serves: {sorted(unknown)}; "
            "refresh the roster with `python -m scripts.refresh_tool_roster` "
            "if the tool is new upstream")
    return frozenset(out)


def _standing_bans() -> frozenset[str]:
    from app.tool_bans import WORKER_AUTOMOD_BAN, WORKER_GRANT_MINT_BAN
    return frozenset(normalize_tool_name(n)
                     for n in (*WORKER_AUTOMOD_BAN, *WORKER_GRANT_MINT_BAN))


# ── The declared sets ────────────────────────────────────────────────────────
#
# One entry per source that has declared one. A source NOT listed here keeps
# today's compile exactly: `RunOptions.allowed_tools` stays None and its deny
# list is the only envelope it gets. That is deliberate — the item's own risk
# rule is one source per round, because the whole failure mode of an allow-list
# is a job that quietly loses a tool it needed.
#
# `session-distill` carries `Bash` because its job is that shell: 344 of the
# 560 tool calls in the 54 retained transcripts are `cd` into
# `~/lloyd-data/sessions` and a `python3 -c` that reads a transcript's own
# JSON, and every one of them is in a run from the last fortnight. `Read` and
# `Grep` are declared beside it because the prompt asks for them; `Bash` is
# declared because the evidence says the distiller would otherwise spend its
# turn rediscovering how to open a file. It stays the tiered-by-command tool
# `app.harness.safety` already treats it as.

SOURCE_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "session-distill": (
        # `FACT_WRITE` is the job: 70 of the 560 calls in its transcripts are
        # `fact_add`, and the distiller exists to make them. Declaring the read
        # half and not the write half is the under-declaration that an
        # allow-list's failure mode is made of — `tests/test_worker_capability_
        # allowlist.py::test_each_source_still_dispatches_the_writes_its_own_
        # job_does[session-distill]` is the node that caught it here.
        "READ_FILES", "READ_MEMORY", "READ_FACTS", "FACT_WRITE", "TURN_STATE",
        "Bash",
    ),
    "youtube-digest": (
        "READ_FILES", "READ_VAULT", "READ_MEMORY", "READ_FACTS", "TURN_STATE",
        "WEB_READ", "FILE_NOTE", "FACT_WRITE", "FILE_BACKLOG",
        "djev_rank", "djev_decide",
    ),
    "deep-research": (
        "READ_VAULT", "READ_FACTS", "TURN_STATE", "READ_MEMORY",
        "WEB_READ", "BROWSER_READ", "FACT_WRITE", "vault_write",
        "backlog_get_task", "graph_explain",
    ),
}

#: The sources whose entire input is text somebody else wrote: a transcript
#: off the internet, a fetched page, another agent's session file. This is the
#: property the sets above are held to — one durable-external reach and the
#: fleet's roster test fails for these by name, so a widening has to be a
#: deliberate edit here and not a set that grew for an unrelated reason.
UNTRUSTED_INGEST_SOURCES: frozenset[str] = frozenset(
    {"session-distill", "youtube-digest", "deep-research"})


def durable_external(names) -> frozenset[str]:
    """Which of `names` are tier-2/3 durable-external by their own name.

    `Bash` is deliberately not answered here: its tier is the command string
    (`policy.tool_tier` asks `safety.bash_command_tier`), so a name-level
    census cannot see it and this function does not pretend to.
    """
    return frozenset(normalize_tool_name(n) for n in names) & (
        frozenset(TIER2_TOOLS) | frozenset(TIER3_TOOLS))


def envelope_for(source: str | None, extra_disallowed=()) -> list[str] | None:
    """This source's allow-list, normalised, or None when it declared none.

    The single compile point for both worker turn shapes — `_worker_run_options`
    hands the result straight to `RunOptions.allowed_tools`, and
    `run_prompt_in_session` carries the same value across the loopback POST to
    `app/routers/turn_options.py`, which is where a session-backed source's
    options are actually built. One derivation, so the set a turn is *told*
    about cannot differ from the one it is dispatched under.

    `extra_disallowed` is subtracted so a caller that forbids a tool for one
    call — `eval/run_scratchpad_ab.py`'s two arms, a source's own `DISALLOWED` —
    narrows the envelope too. Without that the two layers would disagree and
    the deny layer wins at dispatch, which is safe but leaves the turn told it
    may call something it may not.
    """
    declared = SOURCE_CAPABILITIES.get(str(source or ""))
    if declared is None:
        return None
    allowed = set(expand(declared))
    for name in extra_disallowed or ():
        allowed.discard(normalize_tool_name(name))
    return sorted(allowed)


def _validate_declarations() -> None:
    """Expand every declared set once, at import (#2269, clause 4).

    This is why a typo in a capability name cannot become a source that quietly
    lost a tool — the failure mode an allow-list inherits and a deny list never
    had, where the symptom is a job that "just stopped working" rather than a
    boot that says which name it could not resolve. `expand` is the same call
    `envelope_for` makes, so nothing here can pass that the compile would fail.

    What is deliberately NOT checked here is the tier property of
    `UNTRUSTED_INGEST_SOURCES`: `TIER2_TOOLS` is the grant gate's ladder, and a
    tool tiered for *that* reason would then refuse to boot a worker pool whose
    set named it for entirely different ones. That property is a roster test's
    business and the drift report's, where a failure names the source instead
    of the process.
    """
    for source in SOURCE_CAPABILITIES:
        expand(SOURCE_CAPABILITIES[source])


_validate_declarations()
