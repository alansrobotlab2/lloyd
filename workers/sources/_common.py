"""Shared helpers for knowledge-acquisition sources.

All these sources follow the same pattern:
  1. enqueue_if_due scans some watermark / input and enqueues items
  2. execute builds a prompt for the primary model at low vLLM priority (1)
     so interactive chat can preempt it.
  3. response lands under ~/obsidian/pending-research/{source}/{yyyy-mm-dd}/
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from app.paths import VAULT_PENDING_RESEARCH_DIR as STAGING_ROOT

# #529: the worker turn path is where the execution state lives, so the import
# is here and not inside a function. A worker turn used to be one user message
# whose transcript then only ever grew — `loop.py` appends every assistant turn
# and every tool result for the whole life of the turn, and the only releaser,
# pressure-triggered microcompaction, never fires below 0.8 of the truncation
# threshold. `run_prompt_with_run_state` below runs the same job as a sequence
# of steps on a schema-validated `Σ` instead, and the reason the name has to be
# reachable from this module is the item's own premise check: `git log
# -S'RunState'` came back empty because no execution-state object was imported
# anywhere a worker could reach it.
from app.harness.run_state import (  # noqa: E402
    RunState, RunStateResult, run_state_turn,
)

logger = logging.getLogger("lloyd-workers.common")

# How far under the pool's `max_duration_seconds` a turn's own timer sits, so
# the turn always loses its own race rather than being cancelled from outside.
# Larger than `autonomy._POOL_TIMEOUT_MARGIN` (30s) because this path has more
# to do on the way out: it has to reach the backend over HTTP and ask it to
# stop the turn, and then let it finish persisting the transcript.
POOL_TIMEOUT_MARGIN_SECONDS = 60

#: The self-modification loop is not a worker's to drive. Named here rather
#: than inline because two turn paths need it now: `run_prompt_on_primary`
#: bakes it in, and a session-backed source passes it as `extra_disallowed`.
#: `tests/test_automod_hardening.py::test_worker_turns_cannot_drive_the_loop`
#: greps this file for these names, so this is where they live.
WORKER_AUTOMOD_BAN: tuple[str, ...] = (
    "automod_start", "automod_gate", "automod_land",
    "automod_abort", "automod_rollback",
    "automod_vault_land", "automod_vault_revert",
)

#: Minting an authority grant is not a worker's to do either (#534). Same
#: reasoning as the automod ban, named separately because it is enforced twice
#: on purpose: the tool is not advertised on a worker turn, AND the policy hook
#: denies the call if a local model emits it anyway. A turn subject to an
#: authority gate must not be able to write its way out of it — that would be
#: `bypassPermissions` with extra paperwork.
WORKER_GRANT_MINT_BAN: tuple[str, ...] = ("grant_create",)


def build_skill_prompt(skill_text: str, *, job: str, task_block: str) -> str:
    """Render a vault skill plus its concrete task into one worker prompt.

    The sibling of `autonomy._build_task_prompt`, and separate from it for two
    reasons. That one announces "You are executing autonomy task #N", which is
    a lie from a worker source and the kind of lie a model reasons from. And it
    prepends the `[SILENT]` hint, which exists so a scheduled task can decline
    to notify the user — a worker reports through its run record instead, and
    a turn that answers `[SILENT]` here would read as an empty turn.
    """
    return "\n".join([
        f'[SYSTEM: You are running the "{job}" worker job. Follow the skill '
        f'below, applied to the task at the end. Work autonomously and do not '
        f'ask for confirmation.]',
        "",
        skill_text,
        "",
        task_block,
    ])


def parse_confidence(response: str, default: float = 0.5) -> float:
    """Pull `## Confidence\\n<0.0-1.0>` out of a research response.

    One copy. It was pasted into four sources with three subtly different
    bodies, which is three chances for the staging note's confidence field to
    start meaning different things depending on which source wrote it.
    """
    m = re.search(r"confidence[^0-9]*([0-1](?:\.\d+)?)", response, re.IGNORECASE)
    if not m:
        return default
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return default


def turn_timeout_for(source: str, default: float = 3600.0) -> float:
    """A turn's own wall-clock budget, strictly under this source's pool cap."""
    try:
        from workers.sources import get_sources_config
        cap = int(get_sources_config().get(source, {}).get("max_duration_seconds") or 0)
    except Exception:
        cap = 0
    if cap <= 0:
        return float(default)
    return float(max(60, cap - POOL_TIMEOUT_MARGIN_SECONDS))


def staging_dir(source: str) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = STAGING_ROOT / source / today
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_staging_note(
    source: str,
    slug: str,
    body: str,
    confidence: float = 0.5,
    rationale: str = "",
    source_refs: Optional[list[str]] = None,
) -> Path:
    """Write a structured note under pending-research/{source}/{date}/{slug}.md."""
    d = staging_dir(source)
    # Avoid collisions within the same minute.
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    path = d / f"{ts}-{slug}.md"
    fm = {
        "source": source,
        "confidence": round(confidence, 2),
        "review_status": "pending",
        "rationale": rationale,
        "source_refs": source_refs or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    content = f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n\n{body}\n"
    path.write_text(content, encoding="utf-8")
    return path


@dataclass
class TurnResult:
    """What a worker turn produced, and how it ended.

    `text` alone was the old return value, and losing the rest of this is what
    let 225 of the 498 notes under `pending-research/` be written with the body
    `(no response)`. A turn that ends at `max_turns`, or on a tool call, or
    against a wedged engine yields no text — and an empty string is
    indistinguishable from a short answer once the stop reason has been thrown
    away. domain-research then wrote the empty note, ticked the topic off in
    `research-queue.md` so it could never be retried, and returned success.
    Nothing anywhere said a research job had failed.
    """
    text: str = ""
    stop_reason: Optional[str] = None
    num_turns: Optional[int] = None
    usage: dict = field(default_factory=dict)
    #: Set when the turn ran on an execution state (#529). The measurement the
    #: item's acceptance is written against — cumulative prompt tokens, cached
    #: share, estimated prefill seconds, per-step records — is on this object,
    #: because `usage` cannot hold it honestly: there `input_tokens` is the
    #: PEAK single prompt, and the number that matters here is the SUM across
    #: steps. None for every transcript-shaped turn, which is still most of them.
    run_state: Optional[RunStateResult] = None
    #: The transcript this turn wrote, when it wrote one. A caller records it
    #: on its run row so a run record and the session that produced it can be
    #: joined — two records of the same run with nothing connecting them is
    #: what the autonomy path had, and what made a suspect run unreviewable.
    session_id: str = ""

    @property
    def ok(self) -> bool:
        """True when the model actually said something."""
        return bool(self.text.strip())

    def failure_summary(self) -> str:
        return (f"empty response (stop_reason={self.stop_reason}, "
                f"turns={self.num_turns}) — nothing written")


def _worker_run_options(max_turns: int, *, extra_disallowed: Sequence[str] = (),
                        priority: int = 1):
    """Build the `RunOptions` every in-process worker turn runs under, in one place.

    Two turn shapes consume this now — the append-only `run_prompt_on_primary`
    and the state-carried `run_prompt_with_run_state` — and the parts that make
    a worker turn safe must not be able to drift between them: the landing-drain
    check, the tool-override merge, the automod ban, low vLLM priority. A second
    copy is how one of those ends up missing from one path, and the ban has
    already been missing once (see the comment below and
    `tests/test_automod_hardening.py`, which is the only reason it is not
    missing now).
    """
    from app.harness import RunOptions
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from prompt_builder import build_system_prompt
    from autonomy import _get_model_env

    # The landing drain applies to worker turns too. The promoter idles the
    # backend and then restarts it; a worker job that starts in that gap is
    # killed mid-flight, and the connection errors it logs on the way down
    # land inside the observation window and are blamed on the promotion. That
    # is the exact shape of the 2026-09-06 20:14 false positive.
    try:
        from app.routers.automod import drain_active, drain_remaining
        if drain_active():
            raise RuntimeError(
                f"lloyd is landing a code update; not starting a worker turn "
                f"(retry in {drain_remaining():.0f}s)")
    except ImportError:
        pass

    system_prompt = build_system_prompt()

    # `app.config.CONFIG`, not a fresh `yaml.safe_load` of config.yaml. Reading
    # the file directly skips all three things the loader does: `${VAR}`
    # expansion, the `LLOYD_CONFIG_OVERLAY` a canary boots with, and the
    # `data/tool_overrides.yaml` merge. That last one is the live authority for
    # what is switched off — so a tool disabled from the Tools page stayed
    # advertised to every worker turn, which is exactly the drift the override
    # file's warning machinery exists to make impossible.
    from app.config import CONFIG

    disallowed: list[str] = []
    for name, sc in (CONFIG.get("mcp_servers") or {}).items():
        for tname in (sc.get("disabled_tools") or []):
            disallowed.append(f"mcp__{name}__{tname}")

    # A worker job may not drive the self-modification loop. These tools were
    # advertised to every worker prompt, `domain-research` included — and that
    # one reads arbitrary web pages into its context, so the machinery that
    # rewrites production sat one prompt injection away from a source whose
    # entire job is ingesting untrusted text. The backlog triage worker was
    # told not to start a round IN ITS PROMPT, which is not a control.
    for tname in WORKER_AUTOMOD_BAN:
        disallowed.append(tname)
        disallowed.append(f"mcp__lloyd-mcp__{tname}")

    # #534: minting an authority grant is off the menu here too. See the
    # constant — the hook below is the second, load-bearing layer.
    for tname in WORKER_GRANT_MINT_BAN:
        disallowed.append(tname)
        disallowed.append(f"mcp__lloyd-mcp__{tname}")

    disallowed.extend(extra_disallowed)

    # Every non-interactive turn is built here, and until #534 every one of
    # them ran with `hooks=None` — no safety hook, no grant gate, so
    # `safety.py`'s Bash-only patterns never even got a chance to run and
    # nothing at all gated the durable-external surface. The gate takes its
    # scope from `policy.current_scope`, which the pool binds per job.
    from app.harness import HookRegistry
    from app.harness.policy import install_policy_hook
    hooks = HookRegistry()
    install_policy_hook(hooks)

    model_env = _get_model_env("primary")

    return RunOptions(
        model="primary",
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=system_prompt,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
        disallowed_tools=disallowed,
        env=model_env,
        priority=priority,
        hooks=hooks,
    )


async def run_prompt_on_primary(prompt: str, max_turns: int = 20, *,
                                source: str = "worker",
                                title: str = "") -> TurnResult:
    """Dispatch a prompt to the primary model at low vLLM priority.

    **Recorded, not observed.** This path calls `run_query` directly, so it
    has no chat endpoint behind it and therefore no Inner Voice — the observer
    is wired in `app/routers/messages.py` and nowhere else, deliberately, so
    that "how a turn is watched" has one definition. What it *does* now have
    is a session and a transcript: `app/run_recorder.py` persists every event
    as it goes and re-yields it unchanged, so a `gap-fill` or `session-distill`
    run leaves the same readable record an autocode round does.

    That distinction is the whole two-axis model. Recording is cheap and
    universal; observation costs primary capacity per turn and is opt-in per
    source. A source that wants to be watched asks for
    `run_prompt_in_session`, which goes through the chat path.

    `automod_start` still refuses a worker turn and the automod tools are
    still barred below: a round must be *observable*, and a transcript is not
    an observer.
    """
    from app.harness import run_query
    from app.run_recorder import record_events
    from app.sessions_io import create_session, new_background_session_id

    options = _worker_run_options(max_turns)
    session_id = new_background_session_id(source)
    run_id = uuid.uuid4().hex[:12]
    try:
        create_session(session_id, platform="worker", model="primary",
                       title=(title or f"{source} run")[:80], source=source,
                       inner_voice=False, preview=prompt[:60])
        # Same reason as the autonomy path: `turn_id` is what switches on the
        # per-turn change ledger, so what an unattended turn writes to disk is
        # recorded with pre-images and can be reverted.
        options.session_id = session_id
        options.turn_id = run_id
    except Exception as exc:  # noqa: BLE001 — a record is not the run
        logger.warning("could not create session %s for %s: %s",
                       session_id, source, exc)

    messages = [{"role": "user", "content": prompt}]
    out = TurnResult()
    out.session_id = session_id
    chunks: list[str] = []
    async for evt in record_events(run_query(messages, options),
                                   session_id=session_id, turn_id=run_id,
                                   prompt=prompt, model="primary",
                                   source=source):
        if evt["type"] == "text_delta":
            chunks.append(evt.get("text", ""))
        elif evt["type"] == "result":
            # The turn's own account of how it ended. Dropping this event is
            # what made an empty answer look like a successful one.
            out.stop_reason = evt.get("stop_reason")
            out.num_turns = evt.get("num_turns")
            out.usage = evt.get("usage") or {}
            if not chunks and evt.get("response_text"):
                chunks.append(str(evt["response_text"]))
    out.text = "".join(chunks)
    return out


async def run_prompt_with_run_state(
    prompt: str, *,
    job: str,
    state: RunState,
    run_dir: Path,
    skill_text: str = "",
    max_steps: int = 6,
    iterations_per_step: int = 4,
    extra_disallowed: Sequence[str] = (),
    priority: int = 1,
) -> TurnResult:
    """Run one worker job as steps on a schema-validated execution state (#529).

    Same tool policy, same model, same priority as `run_prompt_on_primary` —
    what differs is the substrate. Instead of one prompt whose transcript
    accumulates for the life of the turn, each step gets the skill, the current
    `Σ`, and the latest observation; the model answers with reasoning, a state
    patch and the next action; the patch is validated against `state`'s schema,
    merged with null-deletion, and the segment's transcript is dropped. The
    reasoning goes to `run_dir/state-trace.ndjson` and is never replayed. See
    `app/harness/run_state.py` for why that is a cost story and a correctness
    story at once, and for the trade it makes.

    `prompt` is the concrete task and `skill_text` the immutable specification,
    so a caller with no skill body passes everything as `prompt` and leaves
    `skill_text` empty — the split of `P` is the caller's, not the driver's.

    Raises `RunStateStepError` when a step's patch is invalid on both attempts.
    Deliberate: a caller that wants today's behaviour as a fallback has to catch
    it and say so in code, rather than inherit the transcript by silence.
    """
    result = await run_state_turn(
        job=job,
        skill_text=skill_text,
        task_block=prompt,
        state=state,
        run_dir=run_dir,
        template=_worker_run_options(
            iterations_per_step, extra_disallowed=extra_disallowed,
            priority=priority),
        max_steps=max_steps,
        iterations_per_step=iterations_per_step,
    )
    return TurnResult(
        text=result.text,
        # "stop" only when the model declared the deliverable complete. A run
        # that ran out of steps reports the sentinel the transcript path already
        # uses for the same condition, so callers keep one vocabulary.
        stop_reason="stop" if result.done else "max_turns",
        num_turns=result.num_turns,
        usage={
            # Not `input_tokens`: there that key is the PEAK single prompt, and
            # the number #529's acceptance is written against is the SUM across
            # steps. Under-counting these is what "the transcript is cheap until
            # it isn't" looked like in the dashboard.
            "cumulative_prompt_tokens": result.prompt_tokens,
            "cumulative_cached_tokens": result.cached_tokens,
            "uncached_prompt_tokens": result.uncached_prompt_tokens,
            "finalizer_prompt_tokens": result.finalizer_prompt_tokens,
            "prefill_seconds_estimated": result.prefill_seconds,
            "steps": len(result.steps),
            "iterations": result.iterations,
            "state_rejections": state.rejections,
        },
        run_state=result,
    )


# ---------------------------------------------------------------------------
# Session-backed turns
# ---------------------------------------------------------------------------

class DrainActive(RuntimeError):
    """The backend refused the turn because a landing is in progress."""


class TurnTimeout(RuntimeError):
    """The turn outlived its own budget and was cancelled in the backend."""


class _SSEParser:
    """Incremental SSE parser. One implementation, driven sync or async.

    Event names arrive on their own line ahead of the data line, so a parser
    that is fed one line at a time has to carry that name across calls. The
    first cut of the async reader re-parsed each line in isolation and lost
    it, so every event read as "message".
    """

    def __init__(self) -> None:
        self.event: str | None = None

    def feed(self, line: str) -> list[tuple[str, dict]]:
        line = line.rstrip("\n").rstrip("\r")
        if line.startswith("event:"):
            self.event = line[6:].strip()
            return []
        if not line:
            self.event = None
            return []
        if not line.startswith("data:"):
            return []
        blob = line[5:].strip()
        if not blob:
            return []
        try:
            data = json.loads(blob)
        except ValueError:
            return []
        return [((self.event or data.get("type") or "message"), data)]


def _sse_events(lines):
    """Yield (event, data) pairs from an SSE line iterable. Pure, so testable."""
    parser = _SSEParser()
    for raw in lines:
        yield from parser.feed(raw)


async def _aiter_sse(resp):
    parser = _SSEParser()
    async for line in resp.aiter_lines():
        for pair in parser.feed(line):
            yield pair


def new_worker_session(*, title: str, source: str, model: str = "primary",
                       inner_voice: bool = True) -> str:
    """Create a real session for a worker-driven turn and return its id.

    Named after the source so it is recognisable in the session list and the
    Inner Voice picker, which is the point: a session-backed turn is the only
    kind that leaves a transcript anyone can page back through.

    The file itself is written by `sessions_io.create_session`, which the
    background recorder also uses. This function had its own private copy of
    the session shape, and it already disagreed with the chat path's — it
    wrote `id` where `_save_session_meta` writes `session_id`, and neither
    `last_active` nor `message_count`, so a worker session sorted by its file
    mtime while every chat session sorted by its conversation.
    """
    from app.sessions_io import create_session, new_background_session_id
    return create_session(
        new_background_session_id(source), platform="worker", model=model,
        title=title, source=source, inner_voice=inner_voice)


async def _cancel_session_turn(backend: str, session_id: str) -> bool:
    """Ask the backend to stop the turn running in `session_id`.

    Best effort and never raises: this runs on a path that is already
    failing, and the caller's error is the one worth reporting.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.post(f"{backend}/api/sessions/{session_id}/cancel", json={})
        return r.status_code < 400
    except Exception as exc:
        logger.warning("could not cancel worker turn in %s: %s", session_id, exc)
        return False


async def run_prompt_in_session(prompt: str, *, title: str, source: str,
                                max_turns: int = 60, priority: int = 1,
                                inner_voice: bool = True,
                                timeout_seconds: float | None = None,
                                extra_disallowed: list[str] | None = None,
                                final_schema: dict | None = None,
                                final_schema_prompt: str = "") -> dict:
    """Run one turn through the backend's own chat path, in a real session.

    This is the counterpart to `run_prompt_on_primary`, and the difference is
    the whole reason it exists. `run_prompt_on_primary` calls `run_query`
    directly: no session, no history, no Inner Voice, and the transcript is
    discarded the moment the text has been collected. That is fine for a
    research summary and wrong for anything that judges or changes Lloyd's own
    code, which is work a human needs to be able to review afterwards.

    `app/routers/messages.py` is the ONLY turn path that attaches the observer,
    so rather than copy that wiring into a second place — the codebase has paid
    for second definitions of "due", "healthy" and "tell the human" already —
    this goes through it: a session with `inner_voice` on, then
    `POST /api/message/stream` over loopback, exactly the way the three
    hand-driven automod rounds ran. The turn shows up in the session list, in
    `/health.turns`, and in the Inner Voice tab, and every tool call is
    persisted.

    Returns {text, session_id, stop_reason, num_turns, errors, structured,
    structured_error}. `stop_reason` is the part callers must look at:
    `max_turns` means the model ran out of room, and its final text is not a
    conclusion however finished it reads.

    `final_schema` asks the harness to restate the finished turn as a JSON
    object matching that schema (`app/harness/finalizer.py`). It is honoured
    only for a non-user platform, which every worker session is. `structured`
    is None when it was skipped or failed, and `structured_error` says which —
    a caller keeps its text parser either way.
    Raises `DrainActive` if a landing has the backend draining; the caller
    should skip this run rather than count it.

    **`extra_disallowed` is the only tool control this path has**, and until a
    caller passes it there was none. `run_prompt_on_primary` bakes the automod
    ban into its own `RunOptions`; this path posts to `/api/message/stream`,
    which builds `disallowed_tools` from config plus whatever the body names
    (`app/routers/messages.py::_refresh_disallowed_for_session`). Nothing in
    that endpoint reads `platform`, so a worker session is handed exactly the
    toolbox a chat gets. That is tolerable for backlog triage, which reads only
    this repo, and not for a research turn that fetches arbitrary web pages
    into its context.

    **The budget is wall-clock, and it has to be smaller than the pool's.**
    `timeout_seconds` used to be handed straight to `httpx.Timeout`, where it
    becomes a *per-read* deadline — a stream that produces a token a minute
    never trips it, however long the turn runs. So the only real bound was
    `pool.py`'s `asyncio.wait_for`, and that one cancels the HTTP request
    rather than the turn. The chat path is explicitly built to survive a
    client disconnect ("the consumer keeps running"), which is right for a
    browser tab and wrong here: the pool would record a failure, back off, and
    re-enqueue, and for `autotriage` the retry re-selects the same item
    because no verdict was ever written — a second 90-iteration triage racing
    the first one that never stopped. Defaulting to `turn_timeout_for(source)`
    puts this timer strictly inside the pool's, and expiring it cancels the
    turn in the backend before reporting, the same shape `autonomy.run_task`
    uses against the same pool cap.
    """
    import httpx
    from app.config import service_url

    backend = service_url("backend", "http://127.0.0.1:8080").rstrip("/")
    if timeout_seconds is None:
        timeout_seconds = turn_timeout_for(source)
    session_id = new_worker_session(title=title, source=source, inner_voice=inner_voice)
    payload = {"session_id": session_id, "text": prompt, "model": "primary",
               "priority": int(priority), "max_turns": int(max_turns),
               # The same wall clock this function enforces below, announced to
               # the model. Iterations are not the budget a worker turn dies on:
               # automod round SM_20260909_054722 was killed here with 32 of its
               # 100 iterations unspent, fourteen seconds after committing the
               # work and one `automod_gate` call short of landing it.
               "deadline_seconds": float(timeout_seconds)}
    if extra_disallowed:
        payload["extra_disallowed"] = list(extra_disallowed)
    if final_schema:
        payload["final_schema"] = final_schema
        if final_schema_prompt:
            payload["final_schema_prompt"] = final_schema_prompt

    out: dict = {"text": "", "session_id": session_id, "stop_reason": None,
                 "num_turns": None, "errors": [], "structured": None,
                 "structured_error": ""}

    async def _stream() -> None:
        # Generous per-read timeout: a long tool call legitimately produces no
        # bytes for minutes. The real bound is the wall-clock one below.
        timeout = httpx.Timeout(float(timeout_seconds), connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{backend}/api/message/stream",
                                     json=payload,
                                     headers={"Accept": "text/event-stream"}) as resp:
                if resp.status_code == 503:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    raise DrainActive(body)
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    raise RuntimeError(
                        f"stream endpoint returned {resp.status_code}: {body}")
                async for event, data in _aiter_sse(resp):
                    if event == "error":
                        out["errors"].append(str(data)[:400])
                    elif event == "done":
                        out["text"] = str(data.get("response") or "")
                        out["stop_reason"] = data.get("stop_reason")
                        out["num_turns"] = data.get("num_turns")
                        out["structured"] = data.get("structured")
                        out["structured_error"] = str(data.get("structured_error") or "")
                        break

    try:
        await asyncio.wait_for(_stream(), timeout=float(timeout_seconds))
    except asyncio.TimeoutError:
        cancelled = await _cancel_session_turn(backend, session_id)
        raise TurnTimeout(
            f"worker turn in {session_id} exceeded {timeout_seconds:.0f}s; "
            f"backend cancel {'accepted' if cancelled else 'FAILED'}") from None
    return out
