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
from typing import Optional

from app.paths import SESSIONS_DIR, VAULT_PENDING_RESEARCH_DIR as STAGING_ROOT

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
#: `tests/test_selfmod_hardening.py::test_worker_turns_cannot_drive_the_loop`
#: greps this file for these names, so this is where they live.
WORKER_SELFMOD_BAN: tuple[str, ...] = (
    "selfmod_start", "selfmod_gate", "selfmod_land",
    "selfmod_abort", "selfmod_rollback",
    "selfmod_vault_land", "selfmod_vault_revert",
)


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

    @property
    def ok(self) -> bool:
        """True when the model actually said something."""
        return bool(self.text.strip())

    def failure_summary(self) -> str:
        return (f"empty response (stop_reason={self.stop_reason}, "
                f"turns={self.num_turns}) — nothing written")


async def run_prompt_on_primary(prompt: str, max_turns: int = 20) -> TurnResult:
    """Dispatch a prompt to the primary model at low vLLM priority.

    **No session, therefore no Inner Voice.** `app/routers/messages.py` is the
    only turn path that wires the observer: it needs a session id to read the
    `inner_voice` flag from, to key observations on, and to attach a per-turn
    observer to. A worker turn has none of that, so nothing here is watched and
    nothing lands in the Inner Voice history.

    That is why `selfmod_start` refuses a worker turn and why worker jobs are
    barred from the selfmod tools below: a round must be observable, and this
    path cannot be. If `backlog-selfmod` is ever enabled, its verdicts are
    produced unobserved — acceptable for read-only triage, and the reason the
    fix is to route worker turns through the one IV-capable path rather than to
    copy the observer wiring into a second place.
    """
    from app.harness import run_query, RunOptions
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from prompt_builder import build_system_prompt
    from autonomy import _get_model_env

    # The landing drain applies to worker turns too. The promoter idles the
    # backend and then restarts it; a worker job that starts in that gap is
    # killed mid-flight, and the connection errors it logs on the way down
    # land inside the observation window and are blamed on the promotion. That
    # is the exact shape of the 2026-09-06 20:14 false positive.
    try:
        from app.routers.selfmod import drain_active, drain_remaining
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
    for tname in WORKER_SELFMOD_BAN:
        disallowed.append(tname)
        disallowed.append(f"mcp__lloyd-mcp__{tname}")

    model_env = _get_model_env("primary")

    options = RunOptions(
        model="primary",
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=system_prompt,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
        disallowed_tools=disallowed,
        env=model_env,
        priority=1,
    )

    messages = [{"role": "user", "content": prompt}]
    out = TurnResult()
    chunks: list[str] = []
    async for evt in run_query(messages, options):
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
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_id = f"{ts}_{source.replace('-', '')[:8]}_{uuid.uuid4().hex[:4]}"
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (SESSIONS_DIR / f"{session_id}.json").write_text(json.dumps({
        "id": session_id,
        "title": title,
        "model": model,
        "platform": "worker",
        "source": source,
        "inner_voice": bool(inner_voice),
        "inner_voice_evaluate_user_turns": bool(inner_voice),
        "messages": [],
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2), encoding="utf-8")
    return session_id


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
    hand-driven selfmod rounds ran. The turn shows up in the session list, in
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
    caller passes it there was none. `run_prompt_on_primary` bakes the selfmod
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
    re-enqueue, and for `backlog-selfmod` the retry re-selects the same item
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
               "priority": int(priority), "max_turns": int(max_turns)}
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
