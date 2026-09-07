"""Shared helpers for knowledge-acquisition sources.

All these sources follow the same pattern:
  1. enqueue_if_due scans some watermark / input and enqueues items
  2. execute builds a prompt for the primary model at low vLLM priority (1)
     so interactive chat can preempt it.
  3. response lands under ~/obsidian/pending-research/{source}/{yyyy-mm-dd}/
"""

from __future__ import annotations

import json
import logging
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.paths import LLOYD_HOME, SESSIONS_DIR, VAULT_PENDING_RESEARCH_DIR as STAGING_ROOT

logger = logging.getLogger("lloyd-workers.common")


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


async def run_prompt_on_primary(prompt: str, max_turns: int = 20) -> str:
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
    cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}

    disallowed: list[str] = []
    for name, sc in cfg.get("mcp_servers", {}).items():
        for tname in sc.get("disabled_tools", []):
            disallowed.append(f"mcp__{name}__{tname}")

    # A worker job may not drive the self-modification loop. These tools were
    # advertised to every worker prompt, `domain-research` included — and that
    # one reads arbitrary web pages into its context, so the machinery that
    # rewrites production sat one prompt injection away from a source whose
    # entire job is ingesting untrusted text. The backlog triage worker was
    # told not to start a round IN ITS PROMPT, which is not a control.
    for tname in ("selfmod_start", "selfmod_gate", "selfmod_land",
                  "selfmod_abort", "selfmod_rollback",
                  "selfmod_vault_land", "selfmod_vault_revert"):
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
    final = ""
    async for evt in run_query(messages, options):
        if evt["type"] == "text_delta":
            final += evt.get("text", "")
    return final


# ---------------------------------------------------------------------------
# Session-backed turns
# ---------------------------------------------------------------------------

class DrainActive(RuntimeError):
    """The backend refused the turn because a landing is in progress."""


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


async def run_prompt_in_session(prompt: str, *, title: str, source: str,
                                max_turns: int = 60, priority: int = 1,
                                inner_voice: bool = True,
                                timeout_seconds: float = 3600.0) -> dict:
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

    Returns {text, session_id, stop_reason, num_turns, errors}. `stop_reason`
    is the part callers must look at: `max_turns` means the model ran out of
    room, and its final text is not a conclusion however finished it reads.
    Raises `DrainActive` if a landing has the backend draining; the caller
    should skip this run rather than count it.
    """
    import httpx
    from app.config import service_url

    backend = service_url("backend", "http://127.0.0.1:8080").rstrip("/")
    session_id = new_worker_session(title=title, source=source, inner_voice=inner_voice)
    payload = {"session_id": session_id, "text": prompt, "model": "primary",
               "priority": int(priority), "max_turns": int(max_turns)}

    out: dict = {"text": "", "session_id": session_id, "stop_reason": None,
                 "num_turns": None, "errors": []}
    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{backend}/api/message/stream",
                                 json=payload,
                                 headers={"Accept": "text/event-stream"}) as resp:
            if resp.status_code == 503:
                body = (await resp.aread()).decode("utf-8", "replace")[:300]
                raise DrainActive(body)
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")[:300]
                raise RuntimeError(f"stream endpoint returned {resp.status_code}: {body}")
            async for event, data in _aiter_sse(resp):
                if event == "error":
                    out["errors"].append(str(data)[:400])
                elif event == "done":
                    out["text"] = str(data.get("response") or "")
                    out["stop_reason"] = data.get("stop_reason")
                    out["num_turns"] = data.get("num_turns")
                    break
    return out
