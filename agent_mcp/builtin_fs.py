#!/usr/bin/env python3
"""Lloyd MCP Server: filesystem built-ins (Read, Write, Edit, Grep, Glob).

Recreates the Claude Code built-in tools we lost when we ripped the SDK
out. Names and input schemas mirror the SDK contracts so persisted
session JSON and the persona prompt surfaces keep working unchanged.
Output formatting (e.g. cat -n line prefix on Read) matches too.

The Bash gate is not here: it is `app/harness/safety.py`, installed as a
PreToolUse hook by `install_default_safety_hook` and applied again at
dispatch in `agent_mcp/main.call_tool`. Bash dispatched standalone, outside
the harness and the aggregator, has no gate.

Two refusals stand in front of every write here, in this order: the write
deny-set (`app.harness.protected_paths.write_deny_reason` — a rule about
*where* a write may land, no session required), then the Read-before-write
clobber gate below (a rule about *who looked*). They are separate on purpose:
the config switch that disables the clobber gate must not disable the other.

Mounted into the unified Server("lloyd") via agent_mcp/main.py MODULES.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from mcp.types import Tool

from agent_mcp._shared import (
    ErrorCode,
    _fit_not_found,
    _near_match_hint,
    get_bound_session,
    text_result,
)
from app.atomic_io import commit_lock, write_text_durable
# One byte-ceiling definition for every lane that can rewrite the two loaded
# memory files — this one, `memory_add`/`memory_replace`, `vault_write`, and the
# vault-round validator — with the number and the wording owned by `prompt_surface`.
# Stdlib-only, so it costs this module nothing to import (#1010).
from app.memory_ceiling import memory_write_error

logger = logging.getLogger("lloyd-builtin-fs")

DEFAULT_READ_LINES = 2000
LINE_PREFIX_FMT = "%6d\t%s"
GREP_BIN = "rg"


# ---------------------------------------------------------------------------
# Read tracking — what this session has actually seen of each file
# ---------------------------------------------------------------------------
#
# `Edit` was exact-match against whatever is on disk right now, with no notion
# of whether the model had ever looked at the file. Two failures follow from
# that, and the second is the expensive one:
#
#   * an edit written from memory that happens to match, which is a guess that
#     got lucky;
#   * an edit against content something else rewrote after the Read. The
#     old_string still matches, so the edit *succeeds* — and silently reverts
#     whatever the other writer did. Nothing anywhere reports it.
#
# Keyed by `os.path.realpath` so a symlink and its target are one file, and
# bounded twice: this process serves every session on the box and lives for
# days.
_READ_SESSIONS_MAX = 256
_READ_PATHS_MAX = 2000

# session_id -> realpath -> (mtime_ns, size)
_read_records: "OrderedDict[str, OrderedDict[str, tuple[int, int]]]" = OrderedDict()

# The sync handlers below run on worker threads (`asyncio.to_thread` in
# call_tool), so two sessions really do mutate this concurrently. Never held
# across anything slower than a dict operation.
_state_lock = threading.Lock()


def _stat_key(path: str) -> tuple[int, int] | None:
    """(mtime_ns, size) for `path`, or None if it cannot be stat'd."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _record_seen(session_id: str, real: str, key: tuple[int, int] | None) -> None:
    if not session_id or not real or key is None:
        return
    with _state_lock:
        paths = _read_records.get(session_id)
        if paths is None:
            paths = _read_records[session_id] = OrderedDict()
        _read_records.move_to_end(session_id)
        paths[real] = key
        paths.move_to_end(real)
        while len(paths) > _READ_PATHS_MAX:
            paths.popitem(last=False)
        while len(_read_records) > _READ_SESSIONS_MAX:
            _read_records.popitem(last=False)


def _seen(session_id: str, real: str) -> tuple[int, int] | None:
    with _state_lock:
        paths = _read_records.get(session_id)
        if not paths:
            return None
        rec = paths.get(real)
        if rec is not None:
            paths.move_to_end(real)
            _read_records.move_to_end(session_id)
        return rec


def reset_read_records() -> None:
    """Drop all read tracking. Tests only."""
    with _state_lock:
        _read_records.clear()


def _gates_enabled() -> bool:
    """Read at call time so a test can monkeypatch.setitem the config."""
    try:
        from app.config import CONFIG
        gates = (CONFIG.get("harness") or {}).get("edit_gates") or {}
        return bool(gates.get("enabled", True))
    except Exception:
        return True


def _current_call_id() -> str:
    try:
        from agent_mcp import _task_registry
        return _task_registry.current_call_id.get()
    except Exception:
        return ""


def _ledger_scope(session_id: str):
    """(session, turn) this mutation belongs to, or None for no ledger."""
    try:
        from agent_mcp import _change_ledger, _task_registry
        if not _change_ledger.enabled():
            return None
        return _change_ledger.scope(session_id, _task_registry.current_turn_id.get())
    except Exception:
        logger.warning("change ledger: scope lookup failed", exc_info=True)
        return None


@dataclass
class _Mutation:
    """One in-flight Write or Edit, built on the loop and filled by the thread.

    The handler runs in a worker thread and returns a string; everything the
    loop needs afterwards (what was written, where, whether it succeeded)
    has nowhere else to travel. Later stages — post-edit diagnostics, the
    change ledger — read the same record rather than re-stat'ing the file
    and racing whoever wrote next.
    """

    kind: str                                   # "edit" | "write"
    session_id: str = ""
    gate_on: bool = False
    path: str = ""                              # as the caller gave it, expanded
    real: str = ""                              # os.path.realpath of the above
    existed: bool = False
    pre_bytes: bytes | None = None
    post_text: str = ""
    post_stat: tuple[int, int] | None = None
    ok: bool = False
    # Change-ledger scope and entry, resolved on the loop side and carried
    # through the thread so the snapshot is taken from the same bytes the
    # edit read — re-reading the file here would race whoever writes next.
    scope: tuple[str, str] | None = None
    call_id: str = ""
    ledger_entry: object | None = None


def _ledger_begin(mut: _Mutation, *, op: str) -> None:
    """Open this turn's entry and snapshot the pre-image, before the write.

    Failures are logged and swallowed. An edit that fails because the undo
    bookkeeping failed is strictly worse than an edit with no undo.
    """
    if mut.scope is None:
        return
    try:
        from agent_mcp import _change_ledger
        entry = _change_ledger.begin(
            mut.scope, real=mut.real, path=mut.path, op=op,
            call_id=mut.call_id,
            # Set only when a subagent made the change, so a footer can say
            # which of a turn's writes came from a Task rather than the turn.
            via_session=mut.session_id if mut.session_id.startswith("task:") else "",
        )
        _change_ledger.snapshot_pre(mut.scope, entry, mut.pre_bytes)
        mut.ledger_entry = entry
    except Exception:
        logger.warning("change ledger: begin failed for %s", mut.path, exc_info=True)


def _ledger_commit(mut: _Mutation) -> None:
    if mut.scope is None or mut.ledger_entry is None:
        return
    try:
        from agent_mcp import _change_ledger
        post = _change_ledger.sha256_bytes(mut.post_text.encode("utf-8", "replace"))
        _change_ledger.commit(mut.scope, mut.ledger_entry, post)
    except Exception:
        logger.warning("change ledger: commit failed for %s", mut.path, exc_info=True)


def _protected_path_refusal(mut: _Mutation) -> str | None:
    """The deny-set refusal for this mutation's target, or None to proceed.

    Checked before anything else in `_gate_check`, and outside the `gate_on`
    switch: this is a rule about *where* a write may land, not about whether a
    session looked at the file first, so neither an unbound caller nor the
    config switch that disables the clobber gate may open it. It runs ahead of
    the create early-return for the same reason — otherwise removing a denied
    file and writing it back is a route.

    Fails closed. A deny-set that cannot load is not a deny-set that passed,
    and the alternative — every write silently proceeding while the checker is
    broken — is the exact hole #1049 exists to close. The failure is loud in
    the result and in the log.
    """
    try:
        from app.harness.protected_paths import write_deny_reason
        label = write_deny_reason(mut.real or mut.path)
    except Exception:  # noqa: BLE001 — an unloadable checker must not read as a pass
        logger.exception("protected-path check unavailable for %s", mut.path)
        return json.dumps({
            "error": (f"{'Write' if mut.kind == 'write' else 'Edit'} refused: "
                      f"the protected-path check could not run, so {mut.path} "
                      f"is not being written. Report this rather than "
                      f"working around it."),
            "code": ErrorCode.PROTECTED_PATH,
        })
    if label is None:
        return None
    try:  # one durable row per refusal (app/harness/denial_journal.py); never decides
        from app.harness import denial_journal
        denial_journal.record(guard="protected_write", where="dispatch",
                              session_id=get_bound_session(),
                              tool="Write" if mut.kind == "write" else "Edit",
                              reason=f"protected path ({label})", excerpt=str(mut.path),
                              label=str(label))
    except Exception:  # noqa: BLE001
        pass
    return json.dumps({
        "error": (
            f"{'Write' if mut.kind == 'write' else 'Edit'} refused: {mut.path} "
            f"is protected ({label}). This lane refuses it for every session, and "
            f"so does `vault_write`. Land the change through the route that "
            f"validates it — `automod_vault_land` for vault and prompt surfaces, "
            f"an automod round for code — or ask Alan."
        ),
        "code": ErrorCode.PROTECTED_PATH,
    })


# ── the knowledge/ vocabulary, on this lane too ─────────────────────────────
#
# #872 put the OKF vocabulary in front of `vault_write` (#780: a nightly job
# invented `type: note`, the file landed, and every automod promotion on the box
# blocked until someone found the word). It did not put it in front of THIS lane,
# and this lane writes the same files: on 2026-10-01 an inner-voice turn wrote
# `knowledge/memory/2026-10-01-prefetch-rawspan-no-model-result.md` with
# `type: measurement-note` — a value in no vocabulary, no alias table and no
# revision of anything — through `Write`, and the vault carried one unknown type
# until #2070 found it. A rule that lives on one of two write surfaces is not a
# rule; the surface with the rule is only a detour.
#
# The rule is the strict one — `rejected_document_type`, the read-only half the
# vault lander already asks, not `normalize_document_type`. Not because this lane
# could not rewrite: it could. Because it writes the bytes its caller handed it,
# and a lane that silently edits those bytes behind the caller's back leaves the
# caller's own transcript describing a file that does not exist. The rewriting
# lane is `vault_write`, and the refusal below says so. An alias is therefore
# refused here and rewritten there — the same asymmetry the lander documents, in
# the same direction: a caller that cannot rewrite must not accept.


def _vault_root_real() -> str:
    """`<vault>` realpathed, resolved per call so a relocated HOME or a
    monkeypatched `app.paths.VAULT_ROOT` moves every path guard here with it — the
    reason `protected_paths.protected_roots()` resolves per call too. The
    `app.paths` fallback is the same literal that module is built from."""
    try:
        from app.paths import VAULT_ROOT
        vault = str(VAULT_ROOT)
    except Exception:  # noqa: BLE001 — a checker that cannot import still checks
        vault = os.path.join(os.path.expanduser("~"), "obsidian")
    return os.path.realpath(vault)


def _knowledge_root() -> str:
    """`<vault>/knowledge` realpathed. See `_vault_root_real` for why per call."""
    return os.path.realpath(os.path.join(_vault_root_real(), "knowledge"))


def _knowledge_vocabulary_refusal(target: Path, text: str, tool: str) -> str | None:
    """Refusal for the bytes about to land at `target`, or None to write them.

    The boundary is `agent_mcp/vault.py`'s `_guard_knowledge_type`, restated:
    `knowledge/` and `.md`, nothing else — so a `type:` in `memory/`, in a skill or
    in a scratch file is not this guard's business, and neither is a `.txt` under
    `knowledge/`. What is NOT restated is the rewrite: `vault_write` folds a retired
    spelling onto its canonical value, and this lane corrects nobody's bytes, so an
    alias that would land folded through `vault_write` is refused here rather than
    silently reworded. (Aliases are a one-way change to a caller's text, and this
    tool's contract is that the caller's text lands.)

    Judged on the POST bytes, not the request: an Edit that *repairs* a bad type must
    be allowed through the file that carries it, which is exactly what an earlier
    placement in `_gate_check` could not do — `post_text` is not populated there, and
    for `_edit` the new text only exists under the commit lock. Both call sites below
    price it on `content`/`updated`, beside the identical `memory_write_error` call,
    for the same reason.

    Fails closed on an unimportable vocabulary, in `agent_mcp/vault.py`'s words:
    "a guard whose input cannot be read reports a verdict it cannot justify", and
    failing open here is precisely how an invented value gets back in.
    """
    real = os.path.realpath(str(target))
    if not real.endswith(".md"):
        return None
    knowledge = _knowledge_root()
    if not (real == knowledge or real.startswith(knowledge + os.sep)):
        return None
    try:
        from scripts.vault import okf_taxonomy
    except Exception as exc:  # noqa: BLE001
        return _knowledge_guard_unavailable(
            target, tool, f"scripts.vault.okf_taxonomy is unimportable ({exc})")
    try:
        bad = okf_taxonomy.rejected_document_type(text)
    except Exception as exc:  # noqa: BLE001
        return _knowledge_guard_unavailable(target, tool, str(exc))
    if bad is None:
        return None
    try:  # one durable row per refusal (app/harness/denial_journal.py); never decides
        from app.harness import denial_journal
        denial_journal.record(guard="knowledge_type", where="dispatch",
                              session_id=get_bound_session(), tool=tool,
                              reason=f"invented knowledge type ({bad})",
                              excerpt=str(target), label=str(bad))
    except Exception:  # noqa: BLE001
        pass
    return json.dumps({
        "error": (
            f"{tool} refused: {target} declares `type: {bad}`, which is not one of "
            f"the {len(okf_taxonomy.CANONICAL_TYPES)} canonical knowledge types "
            f"({', '.join(sorted(okf_taxonomy.CANONICAL_TYPES))}). Nothing was "
            f"written. This lane does not rewrite a retired spelling either: "
            f"`vault_write` normalises one to its canonical value, and "
            f"`automod_vault_land` lands a note whose type is already canonical. "
            f"Pick one of the values above."),
        "code": ErrorCode.INVALID_PARAM, "invalid_type": bad, "path": str(target)})


def _knowledge_guard_unavailable(target: Path, tool: str, why: str) -> str:
    """The fail-closed refusal, separate so it is never mistaken for a verdict."""
    logger.error("knowledge/ type guard unavailable for %s: %s", target, why)
    return json.dumps({
        "error": (f"{tool} refused: the knowledge/ type vocabulary could not be "
                  f"read ({why}), so {target} is not being written. Report this "
                  f"rather than working around it."),
        "code": ErrorCode.INTERNAL})


# ── the research route, on this lane too (#3190) ─────────────────────────────
#
# `agent_mcp/vault.py::_guard_research_route` is the same rule on the other lane,
# and #3190's card reached the vault through a file tool, not through
# `vault_write` — so the lane the incident actually used is the one that must not
# accept it. One definition again (`scripts.vault.research_route`, imported at call
# time), and only the PATH shape is lane-specific, for the reason the block below
# gives for the schedule rail: `vault.py` matches the vault-relative string
# `_normalize_vault_path` hands it, this lane matches `os.path.realpath(target)`.


def _vault_rel_for_guard(target_real: str) -> str | None:
    """`target_real` as a vault-relative posix path, or None if it is off-vault.

    The route is a fact about the VAULT's layout, so a `pipeline: research` file
    the caller is writing to /tmp or to a checkout is not this guard's business —
    exactly as the vocabulary guard above leaves everything outside `knowledge/`
    alone. A path that resolves outside the vault returns None, which the caller
    reads as "write it".
    """
    vault = _vault_root_real()
    if not (target_real == vault or target_real.startswith(vault + os.sep)):
        return None
    return os.path.relpath(target_real, vault).replace(os.sep, "/")


def _research_route_refusal(target: Path, text: str, tool: str) -> str | None:
    """Refusal for the bytes about to land at `target`, or None to write them.

    Asks the same predicate `vault_write` asks of the same bytes: refuse a note
    whose front matter declares `pipeline: research` when it is not landing in
    `knowledge/research/`. #3190's card carried exactly that pair at
    `projects/ecep/cpp-adaptive-constraints-predictive-archery-blocks.md`, was
    readable for under two hours, then vanished with no registry row behind it
    (topic 1937 against a registry whose highest id was 129), so nothing that
    checks a research artifact exists on disk ever ran on it.

    Priced on the POST bytes like the vocabulary check beside it — `content` for
    `Write`, `updated` for `Edit` — so an Edit that REMOVES the stray `pipeline:`
    line repairs its way through the file that carries one, and one that adds it
    is refused with nothing written.

    Fails closed on an unimportable route module, in this module's own words above:
    "a guard whose input cannot be read reports a verdict it cannot justify".
    """
    real = os.path.realpath(str(target))
    if not real.endswith(".md"):
        return None
    rel = _vault_rel_for_guard(real)
    if rel is None:
        return None
    try:
        from scripts.vault import research_route
    except Exception as exc:  # noqa: BLE001
        return _route_guard_unavailable(
            target, tool, f"scripts.vault.research_route is unimportable ({exc})")
    try:
        declared = research_route.research_route_refusal(rel, text)
    except Exception as exc:  # noqa: BLE001
        return _route_guard_unavailable(target, tool, str(exc))
    if declared is None:
        return None
    try:  # one durable row per refusal (app/harness/denial_journal.py); never decides
        from app.harness import denial_journal
        denial_journal.record(guard="research_route", where="dispatch",
                              session_id=get_bound_session(), tool=tool,
                              reason=f"off-route `pipeline: {declared}` card",
                              excerpt=str(target), label=str(declared))
    except Exception:  # noqa: BLE001
        pass
    home = os.path.join(_vault_root_real(),
                        *research_route.RESEARCH_NOTE_DIR.split("/"))
    return json.dumps({
        "error": (
            f"{tool} refused: {target} declares `pipeline: {declared}`, which routes "
            f"a research card to {home}/<slug>.md — nothing was written. A card that "
            f"leaves that directory has no registry row, so `research_complete`'s "
            f"on-disk check never runs on it and its loss is unreportable (#3190). "
            f"Write it there — or, if this is not a research card, drop the "
            f"`pipeline: {declared}` key. `vault_write` refuses these bytes at this "
            f"path too."),
        "code": ErrorCode.INVALID_PARAM, "off_route_pipeline": declared,
        "path": str(target)})


def _route_guard_unavailable(target: Path, tool: str, why: str) -> str:
    """The fail-closed refusal, separate so it is never mistaken for a verdict."""
    logger.error("research route guard unavailable for %s: %s", target, why)
    return json.dumps({
        "error": (f"{tool} refused: the research route could not be read ({why}), "
                  f"so {target} is not being written. Report this rather than "
                  f"working around it."),
        "code": ErrorCode.INTERNAL})


# ── the #724 dispatch-field rail, on this lane too (#2516) ──────────────────
#
# #724 wrote the rule for the tool that asks (`autonomy_write_task`), #2190
# extended it to the automod vault round, #2362 extended it to `vault_write`
# after a turn created `autonomy/96-djev-name-prior-probe.md` `status: up_next`
# through that lane and a run started 47 seconds later. This is the third FILE
# lane onto the same directory, and it was the one with nothing in front of it:
# the write deny-set (`app/harness/protected_paths.py`) holds five entries and
# `~/obsidian/autonomy/` — the directory the scheduler reads — is none of them, so
# before this guard a `Write`/`Edit` of a task file passed only the deny-set, the
# memory ceiling and the knowledge vocabulary. An `Edit` flipping `status: draft`
# to `up_next` wrote bytes. (The set's own constant is deliberately not spelled on
# this line or anywhere else in this module:
# `tests/test_protected_paths.py::test_every_lane_that_reaches_a_denied_path_consults_the_deny_set`
# refuses a lane that names it, because a lane that can name the set is a lane that
# can copy it, and the four earlier spellings of "protected" on this box is what
# that rail exists to prevent. Name the module, not its constant.)
#
# The argument is the block above's, unchanged: a rule that lives on one of two
# write surfaces is not a rule, and the surface with the rule is only a detour.
# What is NOT restated below is the rule itself — `policy.front_matter_map`,
# `policy.schedule_value_moves` and `policy.create_arms_a_task` are one
# definition, imported at call time, so widening `SCHEDULE_STATE_FIELDS` widens
# this lane with no edit to this module. That is the claim `app/harness/policy.py`
# makes for "every lane", and until this guard it was true of two of the three.
# Only the PATH shape is lane-specific, and it has to be: `vault.py` matches the
# vault-relative string `_normalize_vault_path` hands it, this lane matches
# `mut.real`, because that is what every other guard here keys on and the one
# value that makes a `./` segment, a `~` prefix and a symlinked parent the same
# file. `test_a_path_reached_by_realpath_is_the_same_file_by_every_lane` pins the
# two predicates agree on extension.
#
# Zero measured traffic through this door — a parse of all 17,097 files under
# `~/lloyd-data/sessions` found 0 `Write`/`Edit` calls naming `obsidian/autonomy/`
# — which is a priority fact, not a verdict one, and the reason this closes a
# completeness gap rather than a live leak. The denial row is what makes the route
# observable from now on: this lane writes no vault audit row
# (`memory/audit/writes.jsonl` is `agent_mcp/vault.py`'s log only), so after this
# lands the only record of anyone reaching a task file here is `denials.jsonl`.


def _autonomy_root() -> str:
    """`<vault>/autonomy` realpathed, resolved per call for the reason
    `_knowledge_root()` gives: a relocated HOME or a monkeypatched
    `app.paths.VAULT_ROOT` has to move this guard with it, and a guard frozen at
    import time is a guard that quietly stopped guarding on a moved box."""
    try:
        from app.paths import VAULT_ROOT
        vault = str(VAULT_ROOT)
    except Exception:  # noqa: BLE001 — a checker that cannot import still checks
        vault = os.path.join(os.path.expanduser("~"), "obsidian")
    return os.path.realpath(os.path.join(vault, "autonomy"))


def _record_schedule_denial(mut: "_Mutation", tool: str, reason: str) -> None:
    """One durable row per refusal (`app/harness/denial_journal.py`); never decides.

    Same `guard` name as the `vault_write` lane's `_record_schedule_refusal`, on
    purpose: it is one rail, and the question owed after landing — did anyone
    reach a task file through a lane that used to be open — is one grep on the
    guard name, with `tool` telling the lanes apart. A journal that is down must
    not become a second verdict, hence the swallowed exception.
    """
    try:
        from app.harness import denial_journal
        denial_journal.record(guard="autonomy_schedule_rail", where="dispatch",
                              session_id=get_bound_session(), tool=tool,
                              reason=reason[:600], excerpt=str(mut.path),
                              label="dispatch-affecting field on an autonomy task file")
    except Exception:  # noqa: BLE001
        pass


def _record_body_denial(mut: "_Mutation", tool: str, reason: str) -> None:
    """One durable row per refused autonomy body (`app/harness/denial_journal.py`).

    One `guard` name shared with the `vault_write` lane's `_record_body_refusal`, on
    the same purpose as `_record_schedule_denial` beside it: it is one rule across the
    free-form lanes, so the owed question — what prose tried to reach a task file, and
    through which lane — is one grep on `guard="autonomy_body_rail"`, with `tool`
    telling the lanes apart. A journal that is down must not become a second verdict.
    """
    try:
        from app.harness import denial_journal
        denial_journal.record(guard="autonomy_body_rail", where="dispatch",
                              session_id=get_bound_session(), tool=tool,
                              reason=f"{mut.path}: {reason}"[:600],
                              excerpt=str(mut.path),
                              label="body orders a tool its built prompt never names")
    except Exception:  # noqa: BLE001
        pass


def _autonomy_body_denial(mut: "_Mutation", post_text: str,
                          tool: str) -> str | None:
    """Refuse a task body ordering a tool its own built prompt never names (#2589).

    The Write/Edit half of the rule the `vault_write` lane enforces in
    `agent_mcp/vault.py::_autonomy_body_refusal`, and the same shipped predicate —
    `agent_mcp._shared.body_undelivered_tools` — is what both call, which is the point:
    the corpus guard, the vault lane and this lane cannot disagree about what an
    offender is, so no body is refused by one and passed by another. The reason it has
    to be on this lane too is the class rule the vault lane's docstring already cites:
    a guard on one of two write surfaces is not a guard.

    Called from the same site as the #2362 dispatch-field rail — before the mutation
    commits — so a refusal here changes no bytes, opens no undo entry and records no
    change-ledger row; the denial journal row is the only thing it writes. Fail-closed
    like the rails beside it: a predicate that cannot run refuses, naming why.
    """
    try:
        from agent_mcp._shared import body_undelivered_tools
        offenders = body_undelivered_tools(post_text)
    except Exception as exc:  # noqa: BLE001 — an unloadable rail is not a rail that passed
        _record_body_denial(mut, tool, f"the predicate itself could not run ({exc})")
        return json.dumps({
            "error": (f"{tool} refused: the autonomy body rail could not run, so "
                      f"{mut.path} is not being written. Report this rather than "
                      "working around it."),
            "code": ErrorCode.PROTECTED_PATH})
    if not offenders:
        return None
    named = ", ".join(f"`{name}`" for name in offenders)
    _record_body_denial(mut, tool, f"body orders {named}")
    return json.dumps({
        "error": (
            f"{tool} refused: {mut.path}'s body orders {named}, and that task's built "
            "prompt never names it. What a run is actually delivered is `silent_hint`, "
            "the SKILL.md its `skill_name` points at and the front-matter `description` "
            "— never the body — so this line reaches no run: it is documentation "
            "claiming an instruction that was never delivered, and every later reader "
            "takes it as one. Put the tool in the `description` (or in that SKILL.md) "
            "if the run is meant to call it, or drop the instruction from the body. "
            "`tests/test_research_doc_claims.py::test_no_task_body_orders_a_tool_its_"
            "prompt_never_names` grades this rule over the whole corpus, so landing "
            "this write would have reddened the next round that touched the file."),
        "code": ErrorCode.PROTECTED_PATH,
        "path": str(mut.path)})


def _autonomy_schedule_refusal(mut: "_Mutation", post_text: str,
                               tool: str) -> str | None:
    """Refusal for task-file bytes about to land through `Write`/`Edit`, or None.

    Priced on the POST bytes at the same site, and for the same reason, as
    `_knowledge_vocabulary_refusal`: for `_edit` the new text only exists under
    the commit lock, so a rule about the bytes about to land cannot be priced on
    the request. `mut.pre_bytes` is that lane's own pre-image — the same bytes the
    change ledger is about to snapshot — so the diff is against what this writer
    actually read, inside one critical section on `mut.real`, which is what phase B
    of the `vault_write` lane does for the same reason. A refusal returns before
    `_ledger_begin`, so a write that never happened opens no undo record, and
    before the `mkdir`, so a refused create leaves no directory behind.

    Four answers, every one of them the rail's:

    * an existing file whose dispatch VALUES do not move — a body edit,
      `description`, an Activity Log line, a re-quoted `scheduled_at` whose parsed
      value is identical — is written;
    * an existing file whose dispatch value does move is refused, naming path,
      field, `old -> new` and the #724 rail;
    * a CREATE is judged only on whether it arms (`status: up_next`), by
      `policy.create_arms_a_task`, the same create branch
      `policy.schedule_fields_changed` uses for the tool call — so a `draft` task
      carrying `skill_name` and `frequency`, the nightly chain's documented
      hand-down shape, still lands;
    * front matter this lane cannot read on either side, or a rail that cannot be
      imported, is a refusal. Failing open there is a rail a caller switches off
      by deleting one `---` line.

    An existing file this lane could not read is a refusal too, and so is one that
    appeared after `mut.existed` was computed: `pre_bytes is None` next to a path
    on disk is a rail that cannot see the value it is supposed to protect.
    """
    real = mut.real or os.path.realpath(mut.path)
    if not real.endswith(".md"):
        return None  # `status` is a backlog item's field too; the scheduler reads these
    root = _autonomy_root()
    if not real.startswith(root + os.sep):
        return None
    try:
        # Read here, at call time, for the same reason the field set is read from
        # the rail and not from a tuple in this file: a lane that cannot load the
        # rule says WHICH fields it could not verify, spelled from the set.
        from app.harness.policy import (DISPATCHING_STATUS, SCHEDULE_STATE_FIELDS,
                                        create_arms_a_task, first_arm_allowed,
                                        front_matter_map, schedule_value_moves,
                                        shown_schedule_value)
    except Exception as exc:  # noqa: BLE001 — an unloadable rail is not a rail that passed
        _record_schedule_denial(mut, tool, f"the rail itself could not run ({exc})")
        return json.dumps({
            "error": (f"{tool} refused: the autonomy schedule rail could not run, so "
                      f"{mut.path} is not being written. Report this rather than "
                      f"working around it."),
            "code": ErrorCode.PROTECTED_PATH})
    held = ", ".join(f"`{f}`" for f in sorted(SCHEDULE_STATE_FIELDS))
    new_fm = front_matter_map(post_text)

    if mut.pre_bytes is None:
        if os.path.exists(real):
            # `_write` sets `pre_bytes` only when its own read succeeds and `_edit`
            # returns before this point when its read fails, so this branch is a
            # genuine read failure or a file that appeared after `mut.existed` was
            # computed. Either way there is no pre-image to diff and no licence to
            # assume the dispatch state nobody has read.
            _record_schedule_denial(mut, tool, "the on-disk task file could not be read")
            return json.dumps({
                "error": (f"{tool} refused: {mut.path} exists and could not be read to "
                          f"compare its dispatch state, so nothing is being written."),
                "code": ErrorCode.PROTECTED_PATH})
        if new_fm is None:
            _record_schedule_denial(mut, tool, "no parseable front matter on a create")
            return json.dumps({
                "error": (f"{tool} refused: {mut.path} would be created with no front "
                          f"matter this rail can read, so none of the dispatch fields "
                          f"the scheduler reads ({held}) can be checked. Refusing "
                          f"rather than guessing."),
                "code": ErrorCode.PROTECTED_PATH})
        if create_arms_a_task(new_fm):
            _record_schedule_denial(mut, tool, f"create with status: {DISPATCHING_STATUS}")
            return json.dumps({
                "error": (f"{tool} refused: {mut.path} would be created `status: "
                          f"{DISPATCHING_STATUS}`, which arms a task — the #724 grant "
                          "rail refuses that unattended on autonomy_write_task and "
                          "this lane holds to the same rule. Create it `status: "
                          "draft` and let a human, or a granted call, arm it."),
                "code": ErrorCode.PROTECTED_PATH})
        # Dispatch state is clean on this create, but the body still came through here,
        # and a create is one of the two ways a bad body arrives (#2589). Ordered after
        # the arming check so a create that is both armed and badly worded is refused
        # for the dangerous half.
        return _autonomy_body_denial(mut, post_text, tool)

    old_fm = front_matter_map(mut.pre_bytes.decode("utf-8", errors="replace"))
    if old_fm is None or new_fm is None:
        which = "incoming" if new_fm is None else "on-disk"
        _record_schedule_denial(mut, tool, "front matter unreadable on one side")
        return json.dumps({
            "error": (f"{tool} refused: the {which} front matter of {mut.path} cannot "
                      f"be read as a mapping, so none of the dispatch fields ({held}) "
                      f"can be checked against it. Repair the front matter rather "
                      f"than writing past it."),
            "code": ErrorCode.PROTECTED_PATH})
    moves = schedule_value_moves(old_fm, new_fm)
    # This lane holds ABSOLUTE paths, and the predicate's path rule is the vault-relative
    # shape the other two lanes pass it; `real` is already known to sit under `_autonomy_root()`
    # (= `<vault>/autonomy`), so spelling it `autonomy/<name>` is one path computation, not a
    # second copy of any condition.
    if not moves or first_arm_allowed("autonomy/" + os.path.relpath(real, root),
                                      old_fm, new_fm):
        # Nothing dispatch-affecting moved, but the post bytes still came through this
        # branch, and #2589's incident is exactly a body rewrite that moved none of them.
        # The dispatch comparison stays first: a write that both moves `status` and orders
        # an undelivered tool is refused for the field, the dangerous half.
        #
        # An allowed FIRST ARM (#3464) falls through here too: whether `draft -> up_next` on
        # a never-run, unparked task may land is `policy.first_arm_allowed`'s answer and
        # nothing in this file decides it, and the body check still runs over the arming
        # write — so a file that both arms a task and orders a tool its prompt never names
        # is still refused for the body.
        return _autonomy_body_denial(mut, post_text, tool)
    field, old, new = moves[0]
    _record_schedule_denial(mut, tool, f"{field}: {old!r} -> {new!r}")
    return json.dumps({
        "error": (
            f"{tool} refused: {mut.path} would move `{field}` "
            f"{shown_schedule_value(old)} -> {shown_schedule_value(new)}"
            + (f" (and {len(moves) - 1} other dispatch field(s): "
               f"{', '.join(f for f, _, _ in moves[1:])}" if len(moves) > 1 else "")
            + " — a dispatch-affecting field, which the #724 grant rail refuses "
              "unattended on autonomy_write_task, and this lane holds to the same "
              f"rule. Land the rest, and let a human or a granted call move "
              f"`{field}`."),
        "code": ErrorCode.PROTECTED_PATH,
        "field": field, "path": str(mut.path)})


def _bench_assertion_refusal(mut: "_Mutation") -> str | None:
    """Refuse a bench task the judge cannot grade, or None to proceed (#2286).

    The `Write`/`Edit` lane of the bench assertion rail — see
    `app/harness/bench_authoring.py` for the property and for why it is a refusal rather
    than a skill instruction. Placed beside `_protected_path_refusal` and BEFORE the
    `mut.gate_on` short-circuit for one reason: everything under the vault is
    out-of-workspace, so a bench path reaches this handler with `gate_on` False, and a
    check placed after that short-circuit would never run for the only paths it exists
    for. `mut.real` is the realpath the `_Mutation` builder already computed, which is
    what makes a symlinked parent and a relative path handed with a `cwd` the same case
    here as on every other lane.

    The key asked is the file's STEM: the bytes live inside `_write_file`/`_edit_file`,
    which run after the gate, so unlike the `vault_write` lane this one cannot see a
    declared frontmatter `id`. The stem is the identity that matters anyway — it is what
    `load_bench_tasks` globs and what the promotion route rewrites an id to — and a file
    written here whose frontmatter id differs is caught one rung later by the coverage
    node, the only place that can see both.
    """
    try:
        from app.harness.bench_authoring import bench_write_defect, record_refusal
        defect = bench_write_defect(mut.real or mut.path,
                                    cwd=(mut.scope[1] if mut.scope else None))
    except Exception as exc:  # noqa: BLE001 — an unloadable rail is not a rail that passed
        logger.exception("bench assertion rail unavailable for %s", mut.path)
        record_refusal("hook", mut.kind, mut.path,
                       f"the rail itself could not run ({exc})")
        return json.dumps({"error": (
            f"Tool call denied: the bench assertion rail could not run ({exc}), so "
            f"{mut.path} is not being written. Report this rather than working around "
            f"it — the alternative is a corpus the judge silently scores two ways."
        )})
    if not defect:
        return None
    record_refusal("hook", mut.kind, mut.path, defect)
    return json.dumps({"error": f"Tool call denied: {defect}"})


def _gate_check(mut: _Mutation) -> str | None:
    """The refusal, as a JSON error string, or None to proceed.

    The write deny-set runs first and unconditionally (see
    `_protected_path_refusal`). What follows is skipped entirely when no
    session is bound: unit tests and legacy callers dispatch straight into
    these handlers with no aggregator context, and a gate that fired there
    would fail `tests/test_mcp_layer.py` rather than protect anything.
    """
    refusal = _protected_path_refusal(mut)
    if refusal is not None:
        return refusal
    refusal = _bench_assertion_refusal(mut)
    if refusal is not None:
        return refusal
    if not mut.gate_on:
        return None
    if mut.kind == "write" and not mut.existed:
        # Creating a file — including through a dangling symlink — needs no
        # prior Read. There is nothing to clobber.
        return None

    rec = _seen(mut.session_id, mut.real)
    if rec is None:
        if mut.kind == "edit":
            return json.dumps({"error": (
                f"Edit refused: {mut.path} has not been Read in this session. "
                f"Read it first (a partial Read with offset/limit is enough), "
                f"then retry the Edit."
            )})
        return json.dumps({"error": (
            f"Write refused: {mut.path} already exists and has not been Read "
            f"in this session. Read it first, then Write it."
        )})

    current = _stat_key(mut.real)
    if current is not None and current != rec:
        if mut.kind == "edit":
            return json.dumps({"error": (
                f"Edit refused: {mut.path} changed on disk since you last Read "
                f"it (size or mtime differ). Read it again, then retry with the "
                f"current content."
            )})
        return json.dumps({"error": (
            f"Write refused: {mut.path} changed on disk since you last Read it. "
            f"Read it again, then retry."
        )})
    return None


def _expand(path: str) -> str:
    """Expand ~ and $VARS so callers can pass `~/obsidian` etc.

    Subprocesses don't expand `~`, and the absolute-path checks below
    reject tilde paths outright. Models reach for `~/...` constantly,
    so normalize at the boundary.
    """
    if not path:
        return path
    return os.path.expanduser(os.path.expandvars(path))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _read(args: dict, session_id: str = "") -> str:
    file_path = _expand(args.get("file_path", ""))
    if not file_path:
        return json.dumps({"error": "file_path is required"})
    if not os.path.isabs(file_path):
        return json.dumps({"error": f"file_path must be absolute, got {file_path!r}"})
    p = Path(file_path)
    if not p.exists():
        return json.dumps({"error": f"file does not exist: {file_path}"})
    if p.is_dir():
        return json.dumps({"error": f"path is a directory, not a file: {file_path}"})

    # Stat BEFORE opening. A write that lands between here and the read makes
    # the recorded key older than the bytes we return, so a later Edit is
    # refused as stale — which is the safe direction. Stat'ing afterwards
    # would record the writer's own key and let that Edit through.
    real = os.path.realpath(file_path)
    seen_key = _stat_key(real)

    # offset is 1-based per the schema (matches the line numbers shown in
    # the output). Default 0 means "start at line 1". Coerce 0 → 1 so the
    # math below is uniform.
    offset = max(1, int(args.get("offset", 0) or 1))
    limit = int(args.get("limit", DEFAULT_READ_LINES) or DEFAULT_READ_LINES)
    if limit <= 0:
        limit = DEFAULT_READ_LINES

    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return json.dumps({"error": f"failed to read {file_path}: {exc}"})

    # A partial Read (offset/limit) records normally: the contract is "you
    # looked at this file at this version", not "you read all of it".
    _record_seen(session_id, real, seen_key)

    if not lines:
        return "<system-reminder>File exists but is empty.</system-reminder>"

    start_idx = offset - 1
    end = min(len(lines), start_idx + limit)
    selected = lines[start_idx:end]
    out_lines = []
    for i, raw in enumerate(selected, start=offset):
        # Match the SDK's behavior: strip the trailing newline before
        # joining so the prefix lines up cleanly.
        text = raw.rstrip("\n")
        out_lines.append(LINE_PREFIX_FMT % (i, text))
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _write_target(mut: "_Mutation", p: Path) -> Path:
    """Where an atomic replace of `p` actually has to land.

    `os.replace` renames over a *name*, so given a symlink it swaps the link
    itself for a regular file and orphans whatever it pointed at. The
    truncate-in-place write this replaced followed the link instead, and for a
    dangling link that meant the write created the link's target
    (`test_write_through_a_dangling_symlink_is_a_create`). Renaming onto the
    resolved path keeps both behaviours: the link stays a link, and a dangling
    one still materialises its target. `mut.real` is that resolution, computed
    once before the lock was taken.
    """
    return Path(mut.real) if os.path.islink(mut.path) else p


def _write(args: dict, mut: _Mutation | None = None) -> str:
    mut = mut if mut is not None else _Mutation(kind="write")
    file_path = _expand(args.get("file_path", ""))
    if not file_path:
        return json.dumps({"error": "file_path is required"})
    if not os.path.isabs(file_path):
        return json.dumps({"error": f"file_path must be absolute, got {file_path!r}"})
    # Same guard as `_edit`, same key-presence test. `content = args.get(
    # "content", "")` above let a `Write` that never sent a body through as a
    # request to truncate: measured on a 280-byte file, `_write({"file_path":
    # p})` returned `'File written: p (0 chars)'` and left 0 bytes on disk,
    # with the undo record opened for a destruction nobody asked for. Presence
    # rather than truthiness because `content: ""` is a real write — creating
    # an empty file is a legitimate call — and because microcompaction keeps
    # the key while replacing a spilled body's value. Refused here, before the
    # gate, the ceiling checks and `_ledger_begin`, so a call that never
    # happened enters no undo record. Pinned by tests/test_required_args.py.
    if "content" not in args:
        return json.dumps({"error": "content is required"})
    content = args["content"]
    p = Path(file_path)
    mut.path = file_path
    mut.real = os.path.realpath(file_path)
    # `exists()` follows symlinks, so a dangling symlink counts as absent and
    # writing through it is a create.
    mut.existed = p.exists()

    refusal = _gate_check(mut)
    if refusal is not None:
        return refusal

    # Pre-image read, ledger snapshot and the replace are one critical section.
    # Read outside the lock and this is the lost update the item is about: the
    # nightly knowledge-write job edits MEMORY.md through this lane while a chat
    # turn's memory_add appends to it, and whichever read second overwrites the
    # other. `p.write_text` also opened O_TRUNC, so a `memory_read` in that
    # window returned an empty file — every writer in the repo shares one commit
    # path here for both reasons. The gate above stays outside: it is a refusal,
    # not part of the commit, and holding the lock across it would put policy
    # latency inside everyone else's wait.
    try:
        with commit_lock(mut.real):
            if mut.existed:
                try:
                    mut.pre_bytes = p.read_bytes()
                except OSError:
                    mut.pre_bytes = None

            # #1010/#507: `lloyd/MEMORY.md` and `lloyd/USER.md` load into every
            # user-platform system prompt and this is the lane the nightly
            # knowledge-write job uses — the one that grew USER.md 48,068 B →
            # 95,302 B in five days with nothing able to refuse it, because
            # `memory_add` was the only guarded writer. A refusal returns before
            # `_ledger_begin`, so a write that never happened enters no undo record.
            ceiling_msg = memory_write_error(p, content)
            if ceiling_msg:
                return json.dumps({"error": ceiling_msg})

            # Same position and same reasoning as the ceiling: a rule about the
            # bytes about to land, priced on `content`, refusing before the undo
            # record opens. This is the lane that put `type: measurement-note`
            # into `knowledge/` on 2026-10-01 (#2070).
            vocab = _knowledge_vocabulary_refusal(p, content, "Write")
            if vocab is not None:
                return vocab

            # The research route, on the same bytes, at the same position, for the
            # same reason (#3190): the card that started it was written by a file
            # tool, not by `vault_write`, so this is the lane the incident used.
            route = _research_route_refusal(p, content, "Write")
            if route is not None:
                return route

            # The #724 dispatch-field rail, on the same bytes and at the same site:
            # `~/obsidian/autonomy/` is in no deny-set, so this was the one lane
            # where a `status: up_next` create or a draft -> up_next move reached a
            # task file with nothing asking. Refuses before the undo record opens
            # and before the `mkdir`, like the vocabulary check beside it.
            schedule = _autonomy_schedule_refusal(mut, content, "Write")
            if schedule is not None:
                return schedule

            _ledger_begin(mut, op="write" if mut.existed else "create")

            try:
                dest = _write_target(mut, p)
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_text_durable(dest, content)
            except OSError as exc:
                return json.dumps({"error": f"failed to write {file_path}: {exc}"})

            mut.post_text = content
            mut.post_stat = _stat_key(mut.real)
            mut.ok = True
            _ledger_commit(mut)
    except TimeoutError as exc:
        return json.dumps({"error": f"failed to write {file_path}: {exc}",
                           "code": "LOCK_TIMEOUT"})
    return f"File written: {file_path} ({len(content)} chars)"


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


def _edit(args: dict, mut: _Mutation | None = None) -> str:
    mut = mut if mut is not None else _Mutation(kind="edit")
    file_path = _expand(args.get("file_path", ""))
    replace_all = bool(args.get("replace_all", False))

    if not file_path:
        return json.dumps({"error": "file_path is required"})
    if not os.path.isabs(file_path):
        return json.dumps({"error": f"file_path must be absolute, got {file_path!r}"})
    # The two string arguments are read only after their keys are proven
    # present, and the test is `key not in args`, never truthiness. Defaulting
    # them to `""` is what turned a forgotten argument into a destructive
    # success: `Edit` with `file_path` + `old_string` and no `new_string` (a
    # mis-sent `content`, which is `Write`'s key, is how it arrives) deleted
    # the matched text and returned `'Edited … (1 replacement)'`, and the same
    # call with `replace_all: true` and no `old_string` walked past the
    # multi-match check — `original.count("")` is len+1 — into
    # `original.replace("", new_string)`, measured on `'abcdef'` as
    # `'-a-b-c-d-e-f-'` over 7 replacements. `""` is a legitimate value for
    # `new_string` (the documented delete path), so presence is the only test
    # that refuses the mistake and keeps that write. It is also the only test
    # compatible with microcompaction: `shrink_assistant_arguments` spills a
    # big body by replacing the *value* with `SHRUNK_ARG_MARKER` and leaves the
    # key in place. No schema layer is involved — `arguments` is validated
    # nowhere on the dispatch path (#1125's owed entry ruled switching that on
    # NO on 2026-10-19), so this handler is the last place that can refuse.
    # Pinned by tests/test_required_args.py.
    for key in ("old_string", "new_string"):
        if key not in args:
            return json.dumps({"error": f"{key} is required"})
    old_string = args["old_string"]
    new_string = args["new_string"]
    if old_string == new_string:
        return json.dumps({"error": "old_string and new_string are identical"})
    p = Path(file_path)
    if not p.exists():
        return json.dumps({"error": f"file does not exist: {file_path}"})

    mut.path = file_path
    mut.real = os.path.realpath(file_path)
    mut.existed = True

    refusal = _gate_check(mut)
    if refusal is not None:
        return refusal

    # Same critical section as `_write`: read, apply, replace, under one lock on
    # the resolved path. This is the lane the nightly knowledge-write job runs on
    # ("Use `Edit` for targeted changes"), so an Edit of lloyd/MEMORY.md used to
    # race a chat turn's memory_add on the same file — 4 writers x 25 appends lost
    # 82 of 100 at base. The old_string check below is inside the lock for the same
    # reason the read is: matching against bytes this lane does not hold is what
    # turns "the edit applied" into "the edit reverted someone else".
    try:
        with commit_lock(mut.real):
            try:
                pre_bytes = p.read_bytes()
            except OSError as exc:
                return json.dumps({"error": f"failed to read {file_path}: {exc}"})
            try:
                original = pre_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Previously this escaped `read_text` as a bare UnicodeDecodeError and
                # left the aggregator, arriving at the model as an MCP exception with
                # no path in it. A binary file is a normal mistake and deserves a
                # normal error.
                return json.dumps({"error": (
                    f"Edit refused: {file_path} is not valid UTF-8 text; Edit only "
                    f"handles text files."
                )})

            count = original.count(old_string)
            if count == 0:
                # `original` is already held under the commit lock, so naming the
                # nearest candidate line costs one string scan and no second read
                # of a file the caller is about to re-Read anyway.
                return json.dumps({"error": _fit_not_found(
                    "old_string not found in file (must match exactly)",
                    _near_match_hint(original, old_string))})
            if count > 1 and not replace_all:
                return json.dumps({
                    "error": f"old_string occurs {count} times — pass replace_all=True or expand the old_string for uniqueness"
                })

            if replace_all:
                updated = original.replace(old_string, new_string)
                replaced = count
            else:
                updated = original.replace(old_string, new_string, 1)
                replaced = 1

            mut.pre_bytes = pre_bytes
            # Same ceiling as `Write`, priced on `updated` — the exact bytes about
            # to land — and allowed to shrink an over-ceiling file, which is how a
            # trim through this same lane stays possible after a refusal.
            ceiling_msg = memory_write_error(p, updated)
            if ceiling_msg:
                return json.dumps({"error": ceiling_msg})
            # Priced on `updated`, not on the file as it stands, so an Edit that
            # REPAIRS a bad `type:` gets through the file that carries one — and
            # one that introduces it is refused with nothing written. `Write`
            # asks the same predicate of `content`, above.
            vocab = _knowledge_vocabulary_refusal(p, updated, "Edit")
            if vocab is not None:
                return vocab
            # Priced on `updated` for the same reason, so an Edit that drops a
            # stray `pipeline: research` line repairs the file that carries one and
            # an Edit that adds one is refused with nothing written (#3190).
            route = _research_route_refusal(p, updated, "Edit")
            if route is not None:
                return route
            # Priced on `updated` for the same reason the vocabulary check above is,
            # and it answers the question the vocabulary check cannot: an Edit
            # arrives as a delta, so `status: draft` -> `up_next` is only visible in
            # the whole file this lane has already assembled under the commit lock.
            schedule = _autonomy_schedule_refusal(mut, updated, "Edit")
            if schedule is not None:
                return schedule
            _ledger_begin(mut, op="edit")

            try:
                write_text_durable(_write_target(mut, p), updated)
            except OSError as exc:
                return json.dumps({"error": f"failed to write {file_path}: {exc}"})

            mut.post_text = updated
            mut.post_stat = _stat_key(mut.real)
            mut.ok = True
            _ledger_commit(mut)
    except TimeoutError as exc:
        return json.dumps({"error": f"failed to edit {file_path}: {exc}",
                           "code": "LOCK_TIMEOUT"})
    return f"Edited {file_path} ({replaced} replacement{'s' if replaced != 1 else ''})"


# ---------------------------------------------------------------------------
# Grep — wraps ripgrep
# ---------------------------------------------------------------------------


async def _grep(args: dict) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return json.dumps({"error": "pattern is required"})
    path = _expand(args.get("path", "")) or os.getcwd()
    output_mode = args.get("output_mode", "files_with_matches")
    head_limit = int(args.get("head_limit", 0) or 0)
    multiline = bool(args.get("multiline", False))
    case_insensitive = bool(args.get("-i", False))
    show_line_numbers = bool(args.get("-n", False))
    after = int(args.get("-A", 0) or 0)
    before = int(args.get("-B", 0) or 0)
    context = int(args.get("-C", 0) or 0)
    glob = args.get("glob", "")
    file_type = args.get("type", "")

    cmd: list[str] = [GREP_BIN]
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    # else "content" mode — ripgrep default
    if case_insensitive:
        cmd.append("-i")
    if show_line_numbers and output_mode == "content":
        cmd.append("-n")
    if multiline:
        cmd.extend(["-U", "--multiline-dotall"])
    if after:
        cmd.extend(["-A", str(after)])
    if before:
        cmd.extend(["-B", str(before)])
    if context:
        cmd.extend(["-C", str(context)])
    if glob:
        cmd.extend(["--glob", glob])
    if file_type:
        cmd.extend(["--type", file_type])
    cmd.extend(["-e", pattern, path])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        return json.dumps({"error": f"{GREP_BIN} (ripgrep) not found on PATH"})
    except Exception as exc:
        return json.dumps({"error": f"grep failed: {exc}"})

    # ripgrep exit codes: 0 = matches, 1 = no matches, 2 = error
    if proc.returncode == 1:
        return "(no matches)"
    if proc.returncode not in (0, 1):
        err = stderr.decode("utf-8", errors="replace").strip()
        return json.dumps({"error": f"ripgrep exited {proc.returncode}: {err}"})

    out = stdout.decode("utf-8", errors="replace")
    if head_limit > 0:
        out = "\n".join(out.splitlines()[:head_limit])
    # Note: results above ~50K chars get spilled to disk by the harness
    # (app.harness.tool_result_spill). We deliberately do NOT truncate
    # here — the spill mechanism preserves the full output on disk so the
    # model can Read it back if it needs more than the preview.
    return out or "(no matches)"


# ---------------------------------------------------------------------------
# Glob — pathlib glob with mtime-desc sort
# ---------------------------------------------------------------------------


def _glob(args: dict) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return json.dumps({"error": "pattern is required"})
    base = _expand(args.get("path", "")) or os.getcwd()
    if not os.path.isabs(base):
        return json.dumps({"error": f"path must be absolute, got {base!r}"})
    base_p = Path(base)
    if not base_p.exists():
        return json.dumps({"error": f"path does not exist: {base}"})

    try:
        matches = list(base_p.glob(pattern))
    except Exception as exc:
        return json.dumps({"error": f"glob failed: {exc}"})

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    matches.sort(key=_mtime, reverse=True)
    if not matches:
        return "(no matches)"
    return "\n".join(str(p) for p in matches)


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


async def list_tools():
    return [
        Tool(
            name="Read",
            description=(
                "Use when you know the exact path and want the contents; to locate a file use Glob or Grep first.\n\n"
                "Read a file from the local filesystem. Returns content with "
                "1-indexed `cat -n`-style line number prefix. Default 2000 "
                "lines per call; pass `offset` and `limit` for larger files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "description": "1-based start line (optional)"},
                    "limit": {"type": "integer", "description": "Max lines to read (default 2000)"},
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="Write",
            description=(
                "Use to create a file or replace one wholesale; to change part of a file you have Read, use Edit instead.\n\n"
                "Write a file to the local filesystem. Creates parent "
                "directories as needed. Creating a new file needs nothing "
                "first; OVERWRITING an existing one requires that you have "
                "Read it in this session and that it has not changed since — "
                "otherwise the write is refused and tells you to Read it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path"},
                    "content": {"type": "string", "description": "File contents"},
                },
                "required": ["file_path", "content"],
            },
        ),
        Tool(
            name="Edit",
            description=(
                "Use to change part of a file you have Read in this session; to replace the whole file use Write instead.\n\n"
                "Replace exact-match text in a file. You must Read the file in "
                "this session first — a partial Read with offset/limit counts — "
                "and the file must not have changed since; the edit is refused "
                "otherwise, which is what stops an edit written from memory "
                "from silently reverting somebody else's write. Errors if "
                "`old_string` is missing or appears multiple times (unless "
                "`replace_all=true`). Your own Write or Edit refreshes the "
                "record, so consecutive edits to one file need only one Read."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to modify",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact text to find, including indentation and "
                            "surrounding lines needed to make it unique in "
                            "the file. Must match byte for byte."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": (
                            "Replacement text. Must differ from old_string; "
                            "pass an empty string to delete the matched text."
                        ),
                    },
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        ),
        Tool(
            name="Grep",
            description=(
                "Use to search file contents by pattern; for file names use Glob, for a known path use Read.\n\n"
                "Search file content with ripgrep. Modes: files_with_matches "
                "(default, list paths), content (lines), count. Supports "
                "case-insensitive (-i), line numbers (-n), context (-A/-B/-C), "
                "glob/type filters, multiline."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Search root (default cwd)"},
                    "glob": {"type": "string", "description": "Glob filter, e.g. '*.py'"},
                    "type": {"type": "string", "description": "ripgrep type, e.g. 'py'"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["files_with_matches", "content", "count"],
                        "description": (
                            "files_with_matches (default) lists matching file "
                            "paths, content shows matching lines, count shows "
                            "per-file match counts."
                        ),
                    },
                    "-i": {"type": "boolean", "description": "Case-insensitive match"},
                    "-n": {
                        "type": "boolean",
                        "description": "Prefix each match with its line number (content mode)",
                    },
                    "-A": {
                        "type": "integer",
                        "description": "Lines of trailing context after each match (content mode)",
                    },
                    "-B": {
                        "type": "integer",
                        "description": "Lines of leading context before each match (content mode)",
                    },
                    "-C": {
                        "type": "integer",
                        "description": "Lines of context on both sides of each match (content mode)",
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": "Let the pattern match across line boundaries (. matches newline)",
                    },
                    "head_limit": {"type": "integer", "description": "Max output lines"},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="Glob",
            description=(
                "Use to find files by name pattern; to search inside files use Grep instead.\n\n"
                "Find files by glob pattern. Returns absolute paths sorted by "
                "modification time, newest first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "e.g. '**/*.py'"},
                    "path": {"type": "string", "description": "Search root (default cwd)"},
                },
                "required": ["pattern"],
            },
        ),
    ]


def _append_diagnostics(text: str, mut: _Mutation) -> str:
    """Append the post-edit blocks to a successful result.

    Two of them, from two stores. `<diagnostics>` is pyflakes on the pre- and
    post-image of *this file*; `<blast_radius>` is the code graph's inbound
    callers for any module-level interface the edit changed, which is the
    part a per-file linter structurally cannot see. Both are advisory.

    Appended only to a success string. `_shared.text_result` sets `isError`
    by sniffing a leading JSON object with an "error" key, so appending to an
    error payload would both break the JSON and flip the flag — a lint
    finding would start reading as a failed edit.

    `call_tool` runs this in a worker thread, never on the event loop
    (#726): pyflakes on two images of up to `MAX_SOURCE_BYTES` each is
    unbudgeted, and the loop also serves SSE chat, voice and the observer.
    The graph half keeps its own `RAIL_BUDGET_S` join inside
    `_edit_diagnostics`, which bounds what an edit waits for it.

    Never raises: the model would rather have an edit with no diagnostics
    than an edit that failed because the linter did.
    """
    try:
        from agent_mcp import _edit_diagnostics as diag
        from agent_mcp import code_graph
        cfg = diag.config()
        if cfg.get("python", True):
            block = diag.python_block(
                mut.path, mut.pre_bytes, mut.post_text,
                int(cfg.get("max_lines", diag.DEFAULT_MAX_LINES)))
            if block:
                text = f"{text}\n\n{block}"
        if cfg.get("blast_radius", True):
            # The live checkout is only a *fallback* root: an automod
            # worktree builds its own graph with `graph_refresh`, and that is
            # the one describing the tree actually being edited.
            blocks = diag.callers_block(
                mut.path, mut.pre_bytes, mut.post_text,
                real=mut.real, fallback_root=str(code_graph.resolve_root(None)))
            if blocks:
                text = f"{text}\n\n{blocks}"
    except Exception:
        logger.warning("edit diagnostics failed for %s", mut.path, exc_info=True)
    return text


async def _append_tsc_hint(text: str, mut: _Mutation) -> str:
    """Queue a debounced whole-project tsc and say so.

    Saying so is not decoration. tsc is whole-project (`include: ["src"]`)
    and ~5 s, so it cannot ride back on the edit — and a model told nothing
    assumes nothing is coming and either re-checks by hand or moves on.
    """
    try:
        from agent_mcp import _tsc_runner
        hint = await _tsc_runner.note_edit(mut.session_id, mut.real)
        if hint:
            return f"{text}\n{hint}"
    except Exception:
        logger.warning("tsc check could not be queued for %s", mut.path, exc_info=True)
    return text


async def call_tool(name: str, arguments: dict):
    # Sync handlers run in a worker thread — this loop also serves SSE chat,
    # voice, and the inner-voice observer; a slow disk read must not stall it.
    session_id = get_bound_session()
    if name == "Read":
        text = await asyncio.to_thread(_read, arguments, session_id)
    elif name in ("Write", "Edit"):
        mut = _Mutation(
            kind="write" if name == "Write" else "edit",
            session_id=session_id,
            # No bound session means no gate: unit tests and legacy callers
            # dispatch straight in, and refusing them protects nothing.
            gate_on=bool(session_id) and _gates_enabled(),
            scope=_ledger_scope(session_id),
            call_id=_current_call_id(),
        )
        handler = _write if name == "Write" else _edit
        text = await asyncio.to_thread(handler, arguments, mut)
        if mut.ok:
            # The writer updates the record, so an Edit immediately after a
            # Write or Edit by the same session is allowed without re-Reading.
            _record_seen(mut.session_id, mut.real, mut.post_stat)
            # Off the loop, like the handler: pyflakes on a large file must
            # not stall every other stream this process serves (#726).
            text = await asyncio.to_thread(_append_diagnostics, text, mut)
            text = await _append_tsc_hint(text, mut)
    elif name == "Grep":
        text = await _grep(arguments)
    elif name == "Glob":
        text = await asyncio.to_thread(_glob, arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return text_result(text)
