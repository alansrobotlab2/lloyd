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
want": the five sets below are the union of (a) every tool name the source's
own prompt or RESULT block instructs the turn to call, and (b) every tool name
the retained session transcripts of that source show it actually calling —
counted over `~/lloyd-data/sessions` on 2026-10-06 (101 `youtube-digest`
sessions/982 calls, 45 `deep-research`/1777, 54 `session-distill`/560), for
`scheduled-task` on 2026-10-09 (566 `*_autonomy_*.json` session files, 13,512
calls; #2471's >5%-of-files rule is the 29-file line), and for `autotriage` on
2026-10-14 (1734 `*_autotriage_*.json`/`.json.gz` files, 32,814 calls, the same
>5%-of-files rule giving an 87-file line; #2671). A tool
is in a set because the job reached for it or its contract names it; `remember`
is the counter-example that shows the measurement working — 19 `deep-research`
calls named it, no such tool is served today, and a deny list could never have
noticed.

**Queue order for the rest (#2471 clause 6, ordered by its owed entry 2).** One
source per round is this roster's own risk rule. `scheduled-task` was the first of
that queue to be declared, because it is the only source with a per-job authority
scope, which is what makes its ceiling widen-able per task rather than per fleet.
`autotriage` is the second (#2671), taking the ruling's own first place, because
it is the largest undeclared surface by volume — 1615 runs in the week to
2026-10-13 against 622 for `owed-check` and 438 for `autocode` — and it reaches
the same 26 durable-external names under the deny union alone. The remaining
eleven —
arch-review, autocode, automod-regression, autoresearch, backlog-cluster,
bench-mine, board-steward, cost-investigate, failure-ledger,
frontend-probe-canary, owed-check — wait for the owed-check ruling (#2269's owed
entry 2) on their order. That eleven is read off the registry, not off #2471's
roster text, which predates `board-steward`, `owed-check` and `cost-investigate`
(#2510, 2026-10-10) existing, listed `autotriage` while it was still undeclared,
and omits `session-distill` from its list of undeclared sources, which has been
declared since #2269.

**The residual risk this does not remove, stated plainly.** An untrusted-ingest
turn that legitimately writes durably (a digest writes a note, files a backlog
item) can still be steered into a *bad durable write*. This narrows what an
injected instruction can do; it does not make injection harder to inject —
that stays `run_injection_canary`'s and the outbound content gate's job.
"""

from __future__ import annotations

import logging

from app.harness.policy import TIER2_TOOLS, TIER3_TOOLS, normalize_tool_name
from app.harness.tool_roster import REGISTERED_TOOLS

logger = logging.getLogger("lloyd.capabilities")

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


#: The ONE standing ban a per-job scope may be licensed for, keyed by the scope
#: prefix that may hold it (#2670, clause 3).
#:
#: `automod_vault_land` is not a loop-control tool: it commits vault bytes
#: already staged on disk, which is what the nightly knowledge jobs' whole job
#: is. It is the single census name the `scheduled-task` ceiling had to exclude
#: for that reason (the 2026-10-09 census: 9.4% of files — 53 of the 566 then
#: retained, every logged call succeeding — the denominator moves with
#: retention, so the SHAPE to check is "the exception exists and names its
#: evidence", not this fraction), and leaving it excluded is what made the
#: ceiling un-wireable into the live turn. The other eight names in
#: `WORKER_AUTOMOD_BAN` are the loop itself — open a round, gate it, land it,
#: abort it, roll it back — and stay refused on every scope, grant row or not:
#: `tests/test_automod_hardening.py::test_an_autonomy_task_scope_is_refused_every_loop_tool_it_is_not_the_exception_of`
#: is the node that keeps this dict from growing by accretion.
#:
#: Reaching an exception requires a LIVE `authority_grants` row for exactly that
#: scope (`live_grant_names` is the only producer of `grant_widening`), so this
#: is not a scope-wide relaxation: with no row, the ban bites identically.
#: `worker:*` scopes are in no key here, so no worker source's envelope can ever
#: name the tool.
SCOPE_BAN_EXCEPTIONS: dict[str, frozenset[str]] = {
    "autonomy-task:": frozenset({"automod_vault_land"}),
}


def scoped_ban_exceptions(scope) -> frozenset[str]:
    """Ban names THIS scope can be licensed for, by prefix — empty for `worker:*`.

    Consulted only by `_widening_names`, never by `expand`: a SOURCE may still
    not declare a standing ban as a capability, because a source's declaration is
    fleet-wide and a licence is one job's. That asymmetry is the whole design —
    `expand` still raises for `automod_vault_land` in a declared set.
    """
    text = str(scope or "")
    out: set[str] = set()
    for prefix, names in SCOPE_BAN_EXCEPTIONS.items():
        if text.startswith(prefix):
            out |= set(names)
    return frozenset(out)


def _widening_names(grant_widening, *, scope: str = "") -> set[str]:
    """Which grant-named tools may join an already-resolved ceiling (#2471).

    Deliberately NOT routed through `expand`: a grant row is runtime data, and
    `expand` raises `UnknownCapability` for a name no server serves and for a
    standing ban. A stale row would then refuse to BOOT the very job the row
    names — the failure `_validate_declarations` chose import-time precisely
    to keep off the run path — so widening is applied to the resolved set
    after `expand`, dropping whatever would have raised. A dropped name is
    not lost authority: dispatch refuses it exactly as it refused it before
    this function existed (the ban via the deny floor and the policy hook, the
    unserved name via P3), which is loud where booting would be silent in the
    other direction.

    The standing-ban drop is also the anti-smuggling rule: a human minted a
    grant row, but `expand`'s "an allow-list that grants `automod_start` is a
    bug" doctrine is a property of what a SOURCE may hold, and a row is not a
    source widening itself — it is one job's named licence, and it cannot put
    a standing worker ban on any turn's catalog.

    `scope` (#2670) is the one crack in that rule, and it is a narrow one: the
    drop still applies to every ban name except
    `SCOPE_BAN_EXCEPTIONS[<scope prefix>]`, which admits exactly
    `automod_vault_land` for an `autonomy-task:<id>` scope holding a live row
    for it. The exception lands the tool in the ceiling ONLY; the deny floor and
    the policy hook beside it are untouched by this module, so a turn that is
    dispatched under the tool still answers to the tier ladder, and a scope with
    no row gets the pre-#2670 compile byte for byte.
    """
    served = registered_tool_names()
    banned = _standing_bans() - scoped_ban_exceptions(scope)
    out: set[str] = set()
    for token in grant_widening or ():
        text = str(token).strip()
        if not text:
            continue
        name = normalize_tool_name(text)
        if name in banned or name not in served:
            continue
        out.add(name)
    return out


def live_grant_names(store, scope: str, *, now=None) -> tuple[str, ...]:
    """Tool patterns of the live grant rows for exactly this scope (#2471).

    `tool_pattern` is matched by EQUALITY by the dispatch gate
    (`app/harness/policy.py`'s `WHERE scope=? AND tool_pattern=?` and the
    `row["tool_pattern"] != tool` comparisons), despite the column's name — so
    widening is a plain name union and nothing here expands a pattern.
    Destination-scoped rows count: they name a tool the scope has been given
    authority over, and the destination axis still applies at dispatch.

    An unreadable store widens NOTHING and logs: the job must still boot on
    its base set, and the grant check at dispatch is what fails closed — this
    function only decides what the turn is told it holds.
    """
    try:
        rows = store.live(scope=scope, now=now)
    except Exception as exc:  # noqa: BLE001 - a dead store is the boot case
        logger.warning("authority_grants unreadable for scope %s (%s): "
                       "widening nothing", scope, exc)
        return ()
    return tuple(str(r.get("tool_pattern") or "").strip()
                 for r in rows if str(r.get("tool_pattern") or "").strip())


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
    "scheduled-task": (
        # Census 2026-10-09 over `~/lloyd-data/sessions/*_autonomy_*.json`:
        # 566 files, 13,512 tool calls; the >5%-of-files rule is the 29-file
        # line (#2471's body quotes 1,089 files/398 turns — that is the
        # pre-retention-sweep denominator; recomputed live per the triage
        # finding, and owed-check rules on it after a week of runs). Above the
        # line: Bash 92.4%, Read 38.7%, backlog_write_task 27.9%, Write
        # 22.1%, Edit 16.4%, vault_write 12.2%, vault_read 8.3%, TodoWrite
        # 7.8%, backlog_get_task 6.9%, backlog_tasks 6.7% — and ONE census
        # name deliberately excluded: `automod_vault_land` (9.4%, 53 files,
        # every logged call succeeding) is a standing worker ban, so `expand`
        # still refuses to declare it, and it enters an envelope only through
        # `SCOPE_BAN_EXCEPTIONS` — a live grant row for one `autonomy-task:<id>`
        # scope, never the base set, so a job with no row is measured against
        # this ceiling without it and the nightly vault-landing jobs are never
        # counted as would-stripped. See the note under this dict.
        #
        # Below the line, by family or prompt, each with its census file
        # count: the sets' sibling legs (Grep 17, Glob 2, vault_search 3,
        # vault_recall 14 — the knowledge jobs' retrieval leg, 115 calls;
        # backlog_boards 24); the durable-internal writes a job's whole job
        # is (fact_add 24 as `FACT_WRITE`, memory_add 12, memory_replace 4,
        # memory_remove 4 — the nightly class-rule jobs write MEMORY.md
        # through these); a job reading its own schedule (`autonomy_get_task`
        # 19); the research-queue job's first moves (`research_stats` 17,
        # `research_list` 16, `research_propose` 14); the skill protocol
        # every worker prompt ships with (`skills_search` 8, `skills_read`
        # 4); web reads (`WEB_READ`, http_fetch 9/http_search 3); and the
        # turn's own read-back (`TURN_STATE` — a compacted turn that lost
        # `recall_observation` is corrupted, not narrowed).
        #
        # No tier-2/3 name is in this base by rule (#2471 clause 2): the 26
        # durable-external names the deny union alone left reachable —
        # `email_empty_junk` through `tasks_update`, and `autonomy_write_task`
        # — are reachable only as a per-JOB widening: a live `authority_grants`
        # row scoped `autonomy-task:<id>` (see `live_grant_names`), never a
        # fleet-wide grant.
        "READ_FILES", "READ_VAULT", "READ_MEMORY", "READ_FACTS", "TURN_STATE",
        "FILE_NOTE", "FILE_BACKLOG", "FACT_WRITE", "WEB_READ",
        "Bash", "Edit", "TodoWrite",
        "memory_add", "memory_replace", "memory_remove",
        "autonomy_get_task",
        "research_stats", "research_list", "research_propose",
        "skills_search", "skills_read",
    ),
    "autotriage": (
        # Census 2026-10-14 over `~/lloyd-data/sessions/*_autotriage_*.json` and
        # the same suffix `.gz`: 1734 files, all parsed, 32,814 tool calls; the
        # >5%-of-files rule is the 87-file line. Above it: Bash 1714 (98.8%),
        # backlog_write_task 1176 (67.8%), Read 455 (26.2%), Grep 191 (11.0%),
        # backlog_get_task 180 (10.4%) — all five are in the base, and
        # `tests/test_worker_capability_allowlist.py::
        # test_autotriage_gets_a_declared_ceiling_its_turn_options_carry` asserts
        # that floor rather than describing it.
        #
        # Below the line, declared anyway with the file count that justified it:
        # `Write` 62 files / 104 calls, and 100 of those 104 wrote a `/tmp/*.py`
        # verification script (the other 4 wrote a scratch probe under
        # `tmp-workdir`, `lloyd-scratch` and `_pipeline`, and none a vault note) —
        # the prompt's own ceiling bullet, "the ONE cheap check that decides the
        # premise", is that script and nothing else,
        # `backlog_tasks` 27, `Scratchpad` 24 as `TURN_STATE`, `automod_status`
        # 21 (`WORKER_AUTOMOD_BAN` names the nine loop DRIVERS and this read-only
        # status tool is not among them; a triage verdict on an item that cites a
        # round needs to read it), `Edit` 13, `Glob` 8 and `skills_search` 6 and
        # `skills_read` 4 (the skill protocol every worker prompt ships with),
        # `vault_read` 6 and `vault_search` 3 as `READ_VAULT`, `memory_read` 3 as
        # `READ_MEMORY`, `vault_recall` as that set's retrieval leg the census
        # never had to reach for. `backlog_boards` and `recall_observation` come
        # in as the same set legs (#2471's rule: a base missing a declared
        # family's siblings corrupts a compacted turn rather than narrowing it).
        #
        # Declared as `Write`/`Edit` by NAME and not as `FILE_NOTE`, on purpose:
        # the set is `{Write, vault_write}`, and `vault_write` is excluded —
        # 0 of 1734 files wrote a vault note, and the fleet's precedent (#2471)
        # keeps the validated vault route automation-only. Taking the set would
        # have silently re-declared, on the source that runs the triage loop, the
        # kind of durable write this round exists to remove; `Edit`, which the set
        # does not contain, is listed because 13 files used it (one of them on a
        # backlog note, the rest on scratch probes).
        #
        # Excluded below the line, recorded rather than dropped quietly because
        # owed-check rules on the width of this base after a week of runs: the
        # autonomy plane (`autonomy_tasks` 6, `autonomy_get_task` 5,
        # `autonomy_health` 1) and the ambient/harness controls
        # (`ambient_decide` 6, `SetGoal` 3, `TodoWrite` 2,
        # `session_inject_context` 1) — reads whose absence makes a turn report
        # it could not look, not a turn that stops filing; and the web
        # (`http_request` 3, `http_fetch` 2), the research queue
        # (`autoresearch_status` 3, `research_stats` 1), the decision engine
        # (`djev_status` 4, `djev_rank` 2) and the code graph
        # (`graph_explain`/`graph_affected` 1 each), the same exclusions #2471
        # recorded for `scheduled-task`. Three census names are not in the served
        # roster at all — `backlog_search` 1, `verdict_block` 1, `Bash_1` 1 —
        # hallucinated spellings a declared set can only refuse, and `expand`
        # would have raised on them at import.
        #
        # No tier-2/3 durable-external name is in this base by rule (#2671
        # clause 2): the 26 the deny union alone left reachable — `email_send`
        # through `tasks_update`, and `autonomy_write_task` — are refused by the
        # envelope on a triage turn. Unlike `scheduled-task` there is no per-job
        # widening crack to close here: `_per_job_grant_widening`
        # (workers/sources/_common.py) returns `()` for any source but
        # `scheduled-task`, and `grant_scope_for` mints an `autonomy-task:<id>`
        # scope for nothing else, so no live grant row can ever add a mail-plane
        # name to this source's ceiling — the base is the whole ceiling.
        "READ_FILES", "READ_VAULT", "READ_MEMORY", "FILE_BACKLOG", "TURN_STATE",
        "Bash", "Write", "Edit", "automod_status",
        "skills_search", "skills_read",
    ),
}

# **How this ceiling reaches the live scheduled-task turn (#2471 → #2670).** A
# scheduled task's live turn is built in `app/autonomy.py::run_task`, which
# predates the envelope machinery and assembles its own `RunOptions` (#534's
# comment says so). Its deny floor is config's `disabled_tools` + the
# grant-mint ban — NOT `WORKER_AUTOMOD_BAN` — which is why 53 census files
# show `automod_vault_land` succeeding there while no worker source may name
# it. That asymmetry was what made this ceiling un-wireable: putting the turn
# under the base alone would have stripped the tool from the nightly
# vault-landing jobs, because a standing ban enters neither a declared set nor
# a widening. `SCOPE_BAN_EXCEPTIONS` above is the ruling that closed it, and
# `app/scheduled_ceiling.compile_surface` is the wiring: `run_task` compiles
# through `envelope_for("scheduled-task", grant_widening=live_grant_names(...))`
# with its own `autonomy-task:<id>` scope, and until seven shadow days say
# otherwise it dispatches `allowed_tools=None` — today's deny floor, unchanged —
# while recording what the ceiling would have refused. See
# `app/scheduled_ceiling.py` for the mode switch and the ledger.

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


def envelope_for(source: str | None, extra_disallowed=(), *,
                 grant_widening=(), scope: str = "") -> list[str] | None:
    """This source's allow-list, normalised, or None when it declared none.

    The single compile point for every turn shape that declares one —
    `_worker_run_options` hands the result straight to
    `RunOptions.allowed_tools`, `run_prompt_in_session` carries the same value
    across the loopback POST to `app/routers/turn_options.py`, which is where a
    session-backed source's options are actually built, and
    `app.scheduled_ceiling.compile_surface` compiles the live scheduled-task
    turn with it (#2670). One derivation, so the set a turn is *told* about
    cannot differ from the one it is dispatched under.

    `extra_disallowed` is subtracted so a caller that forbids a tool for one
    call — `eval/run_scratchpad_ab.py`'s two arms, a source's own `DISALLOWED`,
    the scheduled task's deny floor — narrows the envelope too. Without that the
    two layers would disagree and the deny layer wins at dispatch, which is safe
    but leaves the turn told it may call something it may not.

    `grant_widening` (#2471) is the set of tool names the CALLING job holds a
    live `authority_grants` row for — per job, never across jobs, and read
    from the store by the caller, which is the only layer that knows the run's
    grant scope. It joins the ceiling after `expand` resolves the declared
    base (`_widening_names` says why it must not go through `expand`), and is
    subtracted from by `extra_disallowed` like everything else: a tool the
    operator disabled is not advertised even under a live grant, because the
    MCP server is not serving it and P3 would refuse it.

    `scope` (#2670) is which job's identity that widening was read under, and it
    is only ever read by `_widening_names`, which uses it to decide whether a
    standing ban may be licensed for this scope at all. Omitted — every worker
    compile, the baseline artifact and the guard matrix — it is the empty
    string, no exception applies, and the result is the pre-#2670 compile.
    """
    declared = SOURCE_CAPABILITIES.get(str(source or ""))
    if declared is None:
        return None
    allowed = set(expand(declared)) | _widening_names(grant_widening,
                                                      scope=scope)
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
