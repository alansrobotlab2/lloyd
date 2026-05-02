"""Inner Voice (#345) Stage 3 — multi-persona ensemble + mid-turn drift.

Stage 2 shipped a single persona (`completion_checker`) firing post-loop on
every Inner-Voice ambient turn. Stage 3 grows that into:

  1. **Three personas concurrent** at end-of-turn. Default ensemble is
     `[completion_checker, drift_detector, continuation_drive]`. They run
     under an `asyncio.Semaphore` capped by
     `inner_voice.throughput.max_concurrent_personas` and gathered with
     `asyncio.gather(return_exceptions=True)` so a single persona's HTTP
     blowup doesn't take down the others.

  2. **Real aggregation.** The trivial Stage-2 pass-through is replaced
     with severity_threshold + count_threshold logic from
     `inner_voice.disagreement`. The aggregate `action_chosen` is recorded
     in the event log AND back-propagated onto each Critique's
     `action_taken` field (so SQLite tells the same story).

  3. **Mid-turn drift detection.** New entry point `run_mid_turn_drift_check`
     fires `drift_detector` against the partial response. On disagreement
     above `veto_severity_threshold` the caller in messages.py is expected
     to cancel the turn and inject an ambient nudge — this module just
     surfaces the verdict.

What Stage 3 deliberately does NOT do:
  - End-of-turn intervention dispatch (steer / escalate). That ships in
    Stage 4 alongside consensus termination. Stage 3 records what the
    ensemble would do but doesn't move the agent.
  - Cross-family disposition diversity. Brain 2 still hits the same local
    endpoint as Brain 1; mitigation is purely context-asymmetry techniques
    1+2+3 from the Disagreement Engineering recipe.
  - Lensed past-failure context (technique 3). That lights up in Stage 4
    for `confidence_calibrator` / `hallucination_flag`.

Contract on failure: any unrecoverable error (config missing, persona
file gone, critic call timeout) writes a forensic event and returns
without raising. The chat path must never see an exception.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from app.config import CONFIG
from app import event_log as _event_log
from app.inner_voice import critic as _critic
from app.inner_voice.critic import Critique
from app.paths import SESSIONS_DIR
from usage_store import record_inner_voice_critique

logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Persona prompt loading
# ---------------------------------------------------------------------------

_PERSONAS_DIR = Path("/home/alansrobotlab/obsidian/lloyd/inner_voice/personas")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# (mtime_ns, parsed_meta, body) keyed by persona_name.
_persona_cache: dict[str, tuple[int, dict[str, Any], str]] = {}


def _read_persona_file(persona_name: str) -> tuple[dict[str, Any], str] | None:
    """Read `<personas>/<persona_name>.md` and return (frontmatter, body).

    Returns None if the file is missing or malformed. Caches by mtime so
    edits during a session take effect on the next ambient turn without a
    restart — the spec says budget 5–10 iterations on the prompt.
    """
    path = _PERSONAS_DIR / f"{persona_name}.md"
    if not path.exists():
        logger.warning("persona file missing: %s", path)
        return None
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None

    cached = _persona_cache.get(persona_name)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("persona read failed (%s): %s", persona_name, e)
        return None

    meta: dict[str, Any] = {}
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
            if not isinstance(meta, dict):
                meta = {}
        except yaml.YAMLError as e:
            logger.warning("persona frontmatter YAML error (%s): %s", persona_name, e)
            meta = {}
        body = text[m.end():]

    _persona_cache[persona_name] = (mtime, meta, body)
    return meta, body


# ---------------------------------------------------------------------------
# Transcript window assembly (asymmetric — no thinking)
# ---------------------------------------------------------------------------


def _extract_text_only(content: Any) -> str:
    """Return the human-readable text from a message's `content` field.

    Skips `thinking` blocks unconditionally — Brain 2 must NOT see Brain 1's
    chain of thought (technique 2 from the Disagreement Engineering recipe).
    Tool-use blocks are summarized as `[tool: name]` (no args) so Brain 2
    knows what fired without being anchored to Brain 1's tool-call shape.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            t = (block.get("text") or "").strip()
            if t:
                parts.append(t)
        elif btype == "tool_use":
            name = block.get("name", "?")
            parts.append(f"[tool: {name}]")
        elif btype == "tool_result":
            # Show that a tool returned, but not its content (to keep
            # context tight; Brain 2 can ask via the event log if needed).
            parts.append("[tool_result]")
        # `thinking` and `image` blocks: skip.
    return "\n".join(p for p in parts if p)


def _summarize_tool_calls(tool_calls: list[dict] | None) -> list[str]:
    """Return a list of short tool call descriptors for the current turn.

    Each entry is `<name>(<key=val,...>)` truncated. We include arg keys
    (not values) so the persona prompt's "tool history shows X" rule has
    something to match without the raw payload pollution.
    """
    out: list[str] = []
    for tc in tool_calls or []:
        name = tc.get("name") or tc.get("function", {}).get("name", "?")
        # Normalize MCP-prefixed names for readability.
        short = name.rsplit("__", 1)[-1] if "__" in name else name
        args = tc.get("input") or tc.get("arguments") or tc.get("function", {}).get("arguments", {})
        keys = ""
        if isinstance(args, dict):
            keys = ",".join(sorted(args.keys())[:5])
        elif isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    keys = ",".join(sorted(parsed.keys())[:5])
            except json.JSONDecodeError:
                pass
        out.append(f"{short}({keys})" if keys else short)
    return out


def _load_recent_transcript(
    session_id: str,
    *,
    exclude_turn_id: str | None = None,
    window_turns: int = 5,
) -> list[dict[str, Any]]:
    """Return the last `window_turns` user/assistant pairs from the session.

    Excludes `exclude_turn_id` (the turn currently under critique). Drops
    subliminal / system / tool / silent-ambient entries. Each returned dict
    is `{"role", "text", "turn_id_hint"}` — minimal fields for prompt
    assembly.
    """
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return []
    try:
        data = json.loads(meta_path.read_text())
    except Exception as e:
        logger.warning("transcript load failed (%s): %s", session_id, e)
        return []

    messages = data.get("messages") or []
    if not isinstance(messages, list):
        return []

    convo: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        if msg.get("silent"):
            continue
        # Heuristic match for "current turn": the persisted assistant entry
        # for the active turn carries source != 'user' AND was just appended.
        # We cannot match by turn_id directly because the persisted format
        # uses message ids, not turn ids. Best we can do is drop the LAST
        # assistant message — the consumer always ensures the current turn's
        # assistant text is the last persisted entry by the time this runs.
        text = _extract_text_only(msg.get("content"))
        if not text:
            continue
        convo.append({"role": role, "text": text, "turn_id_hint": msg.get("id", "")})

    # Drop the trailing assistant message — that's the turn under critique.
    if exclude_turn_id and convo and convo[-1]["role"] == "assistant":
        convo = convo[:-1]
    elif convo and convo[-1]["role"] == "assistant":
        # Defensive: if no exclude_turn_id given, still drop the last
        # assistant — it's almost certainly the in-flight turn.
        convo = convo[:-1]

    return convo[-window_turns * 2 :]  # window_turns user+assistant pairs ≈ 2x entries


# ---------------------------------------------------------------------------
# User-prompt assembly for Brain 2
# ---------------------------------------------------------------------------


_TRANSCRIPT_TEXT_CAP = 800   # chars per transcript message
_RESPONSE_TEXT_CAP = 4000    # chars of Brain 1's final response
_PARTIAL_TEXT_CAP = 4000     # chars of Brain 1's partial (mid-turn) response
_TASK_INTENT_CAP = 1500      # chars of frozen task intent


def _truncate(s: str, cap: int) -> str:
    if not s:
        return s
    if len(s) <= cap:
        return s
    return s[: cap - 3] + "..."


# ---------------------------------------------------------------------------
# Stage 5: skill_recall_checker pre-compute
# ---------------------------------------------------------------------------
#
# `skill_recall_checker` (Stage 5) gets a `<matched_skills>` block in its user
# prompt — the top N results from skills_search against the frozen task
# intent. The persona then decides whether the agent ignored a skill that
# matched the task (per its decision rule).
#
# We pre-compute the matches in-process by importing the same scoring
# function the MCP `skills_search` tool uses, so there's no MCP round-trip.
# Caches per-prompt-hash so a repeated call within a session doesn't re-walk
# the skill dirs.

_SKILL_MATCH_CACHE: dict[str, list[tuple[str, str, float]]] = {}
_MAX_SKILL_MATCHES = 3        # how many to surface to the persona
_SKILL_MIN_SCORE_KEEP = 4.0   # below this, prompt shows "(no strong match)"


def _top_matched_skills(query: str, *, k: int = _MAX_SKILL_MATCHES) -> list[tuple[str, str, float]]:
    """Return the top-k (name, description, score) skill matches for `query`.

    Uses `agent_mcp.skills` directly — no MCP subprocess. Falls back to an
    empty list on any import or scoring error so the persona prompt
    degrades gracefully. Cached by raw query string.
    """
    if not query:
        return []
    cached = _SKILL_MATCH_CACHE.get(query)
    if cached is not None:
        return cached

    try:
        from agent_mcp.skills import (
            _iter_skills,        # type: ignore[attr-defined]
            _query_tokens,       # type: ignore[attr-defined]
            _score_skill,        # type: ignore[attr-defined]
        )
    except Exception as e:
        logger.warning("skill_recall: agent_mcp.skills unavailable: %s", e)
        _SKILL_MATCH_CACHE[query] = []
        return []

    try:
        tokens = _query_tokens(query)
        scored: list[tuple[float, str, str]] = []
        for skill in _iter_skills():
            score = _score_skill(skill, tokens, require_metadata_hit=True)
            if score > 0:
                scored.append((score, skill["name"], skill.get("description") or ""))
        scored.sort(key=lambda t: -t[0])
        out = [(name, desc, round(score, 2)) for score, name, desc in scored[:k]]
    except Exception as e:
        logger.warning("skill_recall: skills_search compute failed: %s", e)
        out = []

    _SKILL_MATCH_CACHE[query] = out
    return out


def _format_matched_skills_block(matches: list[tuple[str, str, float]]) -> str:
    """Render the `<matched_skills>` block for the user prompt.

    Each line: `<name> | <description (truncated)> | score <s>`. Returns
    `(empty)` literal when the list is empty so the persona's decision rule
    #1 has a clean signal.
    """
    if not matches:
        return "(empty)"
    rows: list[str] = []
    for name, desc, score in matches:
        if score < _SKILL_MIN_SCORE_KEEP:
            continue
        rows.append(f"{name} | {_truncate(desc, 200)} | score {score}")
    return "\n".join(rows) if rows else "(empty)"


def _build_user_prompt(
    *,
    frozen_task_intent: str,
    response_text: str,
    tool_calls: list[dict],
    transcript: list[dict[str, Any]],
    mode: str = "final",
    matched_skills: list[tuple[str, str, float]] | None = None,
) -> str:
    """Assemble the Brain 2 user-message body.

    Layout (mode='final', the end-of-turn case):
        <task>...</task>
        <recent_transcript>...</recent_transcript>
        <final_response>
        text: ...
        tool_calls: [name1, name2, ...]
        </final_response>

    Layout (mode='partial', the mid-turn case):
        <task>...</task>
        <recent_transcript>...</recent_transcript>
        <partial_response>
        text: ...
        (no tool_calls list — mid-turn drift is text-only by design)
        </partial_response>

    Fixed layout — the persona prompts' examples reference these tag names.
    Changing them invalidates calibration; do not.
    """
    transcript_lines: list[str] = []
    for entry in transcript:
        prefix = "user" if entry["role"] == "user" else "assistant"
        transcript_lines.append(f"[{prefix}] {_truncate(entry['text'], _TRANSCRIPT_TEXT_CAP)}")
    transcript_block = "\n".join(transcript_lines) if transcript_lines else "(no prior turns)"

    if mode == "partial":
        return (
            "<task>\n"
            f"{_truncate(frozen_task_intent or '(no task text)', _TASK_INTENT_CAP)}\n"
            "</task>\n\n"
            "<recent_transcript>\n"
            f"{transcript_block}\n"
            "</recent_transcript>\n\n"
            "<partial_response>\n"
            f"text: {_truncate(response_text or '(empty)', _PARTIAL_TEXT_CAP)}\n"
            "</partial_response>"
        )

    tool_summaries = _summarize_tool_calls(tool_calls)
    tool_str = ", ".join(tool_summaries) if tool_summaries else "(none)"

    # Stage 5: skill_recall_checker reads `<matched_skills>` to decide whether
    # the agent ignored a relevant skill. Other personas don't see this block
    # — adding it unconditionally would just add tokens without signal.
    skills_block = ""
    if matched_skills is not None:
        skills_block = (
            "\n\n<matched_skills>\n"
            f"{_format_matched_skills_block(matched_skills)}\n"
            "</matched_skills>"
        )

    return (
        "<task>\n"
        f"{_truncate(frozen_task_intent or '(no task text)', _TASK_INTENT_CAP)}\n"
        "</task>\n\n"
        "<recent_transcript>\n"
        f"{transcript_block}\n"
        "</recent_transcript>\n\n"
        "<final_response>\n"
        f"text: {_truncate(response_text or '(empty)', _RESPONSE_TEXT_CAP)}\n"
        f"tool_calls: [{tool_str}]\n"
        "</final_response>"
        f"{skills_block}"
    )


# ---------------------------------------------------------------------------
# Throughput guards
# ---------------------------------------------------------------------------


_session_critique_counts: dict[str, int] = {}


def _max_critiques_per_session() -> int:
    iv = CONFIG.get("inner_voice") or {}
    tp = iv.get("throughput") or {}
    return int(tp.get("max_critiques_per_session", 50))


def _max_concurrent_personas() -> int:
    iv = CONFIG.get("inner_voice") or {}
    tp = iv.get("throughput") or {}
    return max(1, int(tp.get("max_concurrent_personas", 5)))


def _under_session_cap(session_id: str, *, n: int = 1) -> bool:
    """Whether `n` more critiques will fit within the session cap."""
    cap = _max_critiques_per_session()
    if cap <= 0:
        return True
    return _session_critique_counts.get(session_id, 0) + n <= cap


def _bump_session_count(session_id: str, *, n: int = 1) -> None:
    _session_critique_counts[session_id] = _session_critique_counts.get(session_id, 0) + n


# ---------------------------------------------------------------------------
# Ensemble selection
# ---------------------------------------------------------------------------


# Tool name sets used by Stage 4 work-type-keyed routing.
# `_DESTRUCTIVE_TOOLS` are tools whose successful invocation cannot be
# trivially reversed without remediation work. They route to
# `safety_critical` regardless of count — one fire is enough to want a
# paranoid review.
_DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({
    # External-world side effects (irreversible without remediation):
    "email_send", "email_delete", "email_forward", "email_reply",
    "discord_send", "discord_send_embed",
    "calendar_create",
    # Memory-graph mutations (vault_write is NOT here — it's research,
    # content creation, routed to research_writing instead):
    "fact_invalidate",
    "memory_remove", "memory_replace",
    # Backlog / autonomy mutations:
    "backlog_write_task",
    "autonomy_write_task", "autonomy_delete_task",
})

# Bash regex patterns that indicate destructive intent. Mirrors the
# `pretooluse_deny` config rules but also catches shapes that just slipped
# past (different argv form, env-var paths, multi-line scripts). Any one
# match is enough to flag.
_BASH_DESTRUCTIVE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\brm\s+-rf?\s+", re.IGNORECASE),
    re.compile(r"\bgit\s+(reset\s+--hard|push\s+.*--force|clean\s+-fd|branch\s+-D)\b"),
    re.compile(r">\s*\S*(?:CONFIG|MEMORY|USER|SOUL)\.md\b"),
    re.compile(r"\bchmod\s+(-R\s+)?[0-9]+\s+/", re.IGNORECASE),
    re.compile(r"\bsudo\s+", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bmkfs", re.IGNORECASE),
)

# Tools that signal a "research" turn: vault writes / many vault searches,
# web fetches, arxiv, deep_research. Threshold-based: 1+ writes OR 3+
# searches OR 1+ web/research fetches.
_RESEARCH_WRITE_TOOLS: frozenset[str] = frozenset({"vault_write"})
_RESEARCH_SEARCH_TOOLS: frozenset[str] = frozenset({
    "vault_search", "vault_recall", "fact_get", "fact_neighbors",
})
_RESEARCH_FETCH_TOOLS: frozenset[str] = frozenset({
    "http_fetch", "http_search", "http_request",
    "arxiv", "deep_research", "deep_dive_research", "websearch",
})


def _bash_command_str(tc: dict) -> str:
    """Extract the Bash `command` string from a tool-call dict, if it's
    a Bash invocation. Empty string for non-Bash or missing input.
    """
    name = (tc.get("name") or tc.get("function", {}).get("name", "")).rsplit("__", 1)[-1]
    if name != "Bash":
        return ""
    args = tc.get("input") or tc.get("arguments") or tc.get("function", {}).get("arguments")
    if isinstance(args, dict):
        return str(args.get("command", "") or "")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return str(parsed.get("command", "") or "")
        except json.JSONDecodeError:
            pass
    return ""


def _select_ensemble_for_turn(
    turn_source: str,
    frozen_task_intent: str,
    *,
    tool_calls: list[dict] | None = None,
) -> tuple[str, list[str], str]:
    """Pick which persona ensemble to fire for this turn.

    Returns `(ensemble_name, personas, rationale)`. Routing priority:

      1. `safety_critical` — destructive Bash patterns OR ≥1 mutation
         tool from `_DESTRUCTIVE_TOOLS`. Highest priority because the
         blast radius justifies the extra Brain 2 cost.
      2. `research_writing` — vault writes, many vault searches, or web/
         arxiv fetches. Catches knowledge-build turns where
         `hallucination_flag` is the most informative voice.
      3. `code_writing` — ≥2 edit-shaped tool calls (Edit, Write,
         MultiEdit). Existing Stage 3 heuristic.
      4. `autonomy_default` — fallback. The everyman ensemble.

    The rationale string lands in `inner_voice.ensemble_selected` events
    so post-hoc analysis can tell *why* a particular ensemble fired.
    """
    iv = CONFIG.get("inner_voice") or {}
    ens = iv.get("ensemble") or {}
    sets = ens.get("sets") or {}
    default_name = ens.get("default") or "autonomy_default"

    tcs = list(tool_calls or [])
    tool_names = [
        (tc.get("name") or tc.get("function", {}).get("name", "")).rsplit("__", 1)[-1]
        for tc in tcs
    ]

    # Stage 4: safety_critical signal — destructive Bash OR mutation tool.
    safety_bash_hits: list[str] = []
    for tc in tcs:
        cmd = _bash_command_str(tc)
        if not cmd:
            continue
        for pat in _BASH_DESTRUCTIVE_PATTERNS:
            if pat.search(cmd):
                safety_bash_hits.append(pat.pattern)
                break
    safety_tool_hits = [n for n in tool_names if n in _DESTRUCTIVE_TOOLS]

    # Stage 4: research signal — writes, search count, web fetches.
    research_writes = sum(1 for n in tool_names if n in _RESEARCH_WRITE_TOOLS)
    research_searches = sum(1 for n in tool_names if n in _RESEARCH_SEARCH_TOOLS)
    research_fetches = sum(1 for n in tool_names if n in _RESEARCH_FETCH_TOOLS)
    research_signal = (
        research_writes >= 1
        or research_searches >= 3
        or research_fetches >= 1
    )

    # Existing code_writing signal — preserved.
    code_signals = sum(1 for n in tool_names if n in {"Edit", "Write", "MultiEdit"})

    rationale = "default ensemble; no work-type signal triggered"
    name = default_name

    # Priority: safety > research > code > default. The first signal that
    # lights up wins so we never under-route a destructive turn into a
    # gentler ensemble.
    if (safety_bash_hits or safety_tool_hits) and "safety_critical" in sets:
        name = "safety_critical"
        bits: list[str] = []
        if safety_bash_hits:
            bits.append(f"bash patterns matched: {safety_bash_hits[:3]}")
        if safety_tool_hits:
            bits.append(f"mutation tools fired: {safety_tool_hits[:5]}")
        rationale = "safety_critical: " + "; ".join(bits)
    elif research_signal and "research_writing" in sets:
        name = "research_writing"
        rationale = (
            f"research_writing: writes={research_writes} searches={research_searches} "
            f"fetches={research_fetches}"
        )
    elif code_signals >= 2:
        name = "code_writing" if "code_writing" in sets else default_name
        rationale = f"code_writing: {code_signals} edit-shaped tool calls in turn history"

    personas = list(sets.get(name) or sets.get("autonomy_default") or ["completion_checker"])
    # Stage 3 ships only personas whose prompt files exist on disk. Drop
    # any that haven't been written yet (e.g. Stage-4 personas referenced
    # in `safety_critical`). This keeps config.yaml forward-looking
    # without crashing on missing files.
    available: list[str] = []
    for p in personas:
        if (_PERSONAS_DIR / f"{p}.md").exists():
            available.append(p)
        else:
            logger.info("ensemble: dropping persona %r (prompt not yet written)", p)
    if not available:
        # Last-ditch fallback to completion_checker (Stage 2's known-good).
        available = ["completion_checker"]
        rationale = f"{rationale}; ALL personas missing → fallback to completion_checker"
    return name, available, rationale


# ---------------------------------------------------------------------------
# Public entry points — wired from messages.py
# ---------------------------------------------------------------------------


# Type alias for the SSE emit hook. Producers in messages.py pass a callback
# that pushes onto the active turn's event queue.
EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


async def run_post_loop_critique(
    *,
    session_id: str,
    turn_id: str,
    turn_source: str,
    frozen_task_intent: str,
    response_text: str,
    tool_calls: list[dict],
    emit_sse: EmitFn | None = None,
) -> list[Critique]:
    """Run the end-of-turn ensemble (3 personas concurrent) on a finished
    ambient turn. Returns the list of critiques (in deterministic order).

    Spawned via `asyncio.ensure_future` from the ResultMessage branch in
    `_run_turn`. Best-effort. Catches every exception on its own — never
    raises into the consumer.

    Stage 3 rules:
      - `asyncio.gather(return_exceptions=True)` so one persona's HTTP
        failure doesn't kill the others.
      - Concurrency cap from `inner_voice.throughput.max_concurrent_personas`
        via a Semaphore.
      - `_aggregate` returns the threshold-driven action_chosen which is
        also written back onto each Critique's `action_taken` field for
        SQLite. End-of-turn dispatch (steer / escalate) is Stage 4.
    """
    out: list[Critique] = []
    if not _is_enabled():
        return out

    ensemble_name, personas, rationale = _select_ensemble_for_turn(
        turn_source, frozen_task_intent, tool_calls=tool_calls,
    )

    # Cap-budget check — refuse to start the ensemble if we'd blow the
    # session's critique budget. Logged as a single skip event covering
    # the whole ensemble (cleaner than one event per persona).
    if not _under_session_cap(session_id, n=len(personas)):
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_skipped",
                {
                    "reason": "session_critique_cap_reached",
                    "cap": _max_critiques_per_session(),
                    "skipped_personas": personas,
                    "ensemble_name": ensemble_name,
                },
                turn_id=turn_id,
            )
        except Exception:
            pass
        return out

    try:
        if emit_sse is not None:
            await emit_sse("inner_voice_observation_state", {"state": "critiquing"})
            await emit_sse(
                "inner_voice_ensemble_change",
                {"ensemble_name": ensemble_name, "personas": personas},
            )
    except Exception as e:
        logger.warning("inner_voice ensemble emit failed: %s", e)

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.ensemble_selected",
            {
                "ensemble_name": ensemble_name,
                "personas": personas,
                "selection_rationale": rationale,
                "turn_source": turn_source,
            },
            turn_id=turn_id,
        )
    except Exception:
        pass

    transcript_window_turns = int(_critic._critic_cfg().get("transcript_window_turns", 5))
    transcript = _load_recent_transcript(
        session_id,
        exclude_turn_id=turn_id,
        window_turns=transcript_window_turns,
    )
    base_user_prompt = _build_user_prompt(
        frozen_task_intent=frozen_task_intent,
        response_text=response_text,
        tool_calls=tool_calls,
        transcript=transcript,
        mode="final",
    )

    # Stage 5: pre-compute matched skills once if any persona in the
    # ensemble needs them. `skill_recall_checker` is currently the only
    # consumer; the cache key is the task intent so future personas
    # reading the same block share the lookup.
    matched_skills: list[tuple[str, str, float]] | None = None
    if "skill_recall_checker" in personas:
        try:
            matched_skills = _top_matched_skills(frozen_task_intent or "")
            _event_log.log_event(
                session_id,
                "inner_voice.skill_recall_matches",
                {
                    "match_count": len(matched_skills),
                    "matches": [
                        {"name": n, "score": s} for n, _d, s in matched_skills
                    ],
                },
                turn_id=turn_id,
            )
        except Exception as e:
            logger.warning("skill_recall pre-compute failed: %s", e)
            matched_skills = []

    skill_recall_user_prompt = (
        _build_user_prompt(
            frozen_task_intent=frozen_task_intent,
            response_text=response_text,
            tool_calls=tool_calls,
            transcript=transcript,
            mode="final",
            matched_skills=matched_skills or [],
        )
        if matched_skills is not None
        else base_user_prompt
    )

    def _prompt_for(persona_name: str) -> str:
        return (
            skill_recall_user_prompt
            if persona_name == "skill_recall_checker"
            else base_user_prompt
        )

    # ─── Stage 3: concurrent fan-out ────────────────────────────────────
    # Semaphore bounds in-flight Brain 2 calls. Cap is per-process, not
    # per-turn — if multiple turns are concurrent on the same process
    # the cap still holds. `gather(return_exceptions=True)` so a single
    # persona's blowup doesn't take down its siblings.
    sem = asyncio.Semaphore(_max_concurrent_personas())

    async def _sem_run(p: str) -> Critique | None:
        async with sem:
            return await _run_one_persona(
                session_id=session_id,
                turn_id=turn_id,
                persona_name=p,
                user_prompt=_prompt_for(p),
                response_excerpt=response_text or "",
            )

    raw_results = await asyncio.gather(
        *(_sem_run(p) for p in personas),
        return_exceptions=True,
    )

    # Drop exceptions/None; preserve persona-list order for downstream
    # determinism. Log each exception so they're not swallowed silently.
    critiques: list[Critique] = []
    for p, r in zip(personas, raw_results):
        if isinstance(r, Exception):
            logger.warning("inner_voice persona %s raised: %s", p, r)
            try:
                _event_log.log_event(
                    session_id,
                    "inner_voice.persona_skipped",
                    {"reason": "persona_invocation_exception",
                     "persona": p, "error": f"{type(r).__name__}: {r}"},
                    turn_id=turn_id,
                )
            except Exception:
                pass
            continue
        if r is None:
            continue
        critiques.append(r)

    # Bump the per-session counter only for personas that actually ran.
    if critiques:
        _bump_session_count(session_id, n=len(critiques))

    # ─── Aggregate + back-propagate action_taken ────────────────────────
    agg = _aggregate(critiques)
    for c in critiques:
        c.action_taken = _resolve_per_critique_action(c, agg)

    # Persist all critiques (after action_taken is finalized).
    for c in critiques:
        try:
            _persist_critique(session_id=session_id, turn_id=turn_id, critique=c)
        except Exception as e:
            logger.warning("inner_voice critique persist failed (%s): %s", c.persona, e)
        if emit_sse is not None:
            try:
                await _emit_critique_sse(emit_sse, c, turn_id=turn_id)
            except Exception as e:
                logger.warning("inner_voice critique emit failed: %s", e)

    # ─── Aggregation event log ──────────────────────────────────────────
    try:
        iv_dis = (CONFIG.get("inner_voice") or {}).get("disagreement") or {}
        _event_log.log_event(
            session_id,
            "inner_voice.aggregation_decision",
            {
                "ensemble_name": ensemble_name,
                "personas_invoked": len(critiques),
                "personas_disagreed": agg["disagree_count"],
                "severity_max": agg["severity_max"],
                "severity_mean": agg["severity_mean"],
                "action_chosen": agg["action_chosen"],
                "thresholds": {
                    "severity_threshold": float(iv_dis.get("severity_threshold", 0.6)),
                    "count_threshold": int(iv_dis.get("count_threshold", 2)),
                    "veto_severity_threshold": float(iv_dis.get("veto_severity_threshold", 0.85)),
                },
                "rationale": agg["rationale"],
                "stage": "stage3_post_loop",
            },
            turn_id=turn_id,
        )
    except Exception:
        pass

    try:
        if emit_sse is not None:
            await emit_sse("inner_voice_observation_state", {"state": "observing"})
    except Exception:
        pass

    out.extend(critiques)
    return out


async def run_mid_turn_drift_check(
    *,
    session_id: str,
    turn_id: str,
    frozen_task_intent: str,
    partial_response: str,
    stream_position_chars: int,
    delta_index: int,
    emit_sse: EmitFn | None = None,
) -> Critique | None:
    """Fire `drift_detector` against a partial-response stream sample.

    Returns the Critique on every fire (success, agreement, or error) so
    the caller in messages.py can branch on `severity` / `error`. Returns
    None only if config-disabled or session cap reached.

    Caller is responsible for the cancel + ambient-inject dispatch when
    the verdict is `disagrees=True` and `severity >= veto_severity_threshold`.
    This function only:
      - assembles the `<partial_response>` user prompt
      - invokes drift_detector via `_run_one_persona`
      - persists the Critique with action_taken inferred from the verdict
      - logs the `inner_voice.aggregation_decision` event
      - emits the SSE notification
    """
    if not _is_enabled():
        return None
    if not _is_mid_turn_drift_enabled():
        return None
    if not _under_session_cap(session_id, n=1):
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_skipped",
                {"reason": "session_critique_cap_reached",
                 "cap": _max_critiques_per_session(),
                 "skipped_personas": ["drift_detector"],
                 "stage": "mid_turn"},
                turn_id=turn_id,
            )
        except Exception:
            pass
        return None

    try:
        if emit_sse is not None:
            await emit_sse(
                "inner_voice_observation_state",
                {"state": "critiquing", "stage": "mid_turn"},
            )
    except Exception:
        pass

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.mid_turn_check_started",
            {
                "persona": "drift_detector",
                "stream_position_chars": stream_position_chars,
                "delta_index": delta_index,
                "partial_chars": len(partial_response or ""),
            },
            turn_id=turn_id,
        )
    except Exception:
        pass

    transcript_window_turns = int(_critic._critic_cfg().get("transcript_window_turns", 5))
    transcript = _load_recent_transcript(
        session_id,
        exclude_turn_id=turn_id,
        window_turns=transcript_window_turns,
    )
    user_prompt = _build_user_prompt(
        frozen_task_intent=frozen_task_intent,
        response_text=partial_response,
        tool_calls=[],   # mid-turn checks are text-only
        transcript=transcript,
        mode="partial",
    )

    critique = await _run_one_persona(
        session_id=session_id,
        turn_id=turn_id,
        persona_name="drift_detector",
        user_prompt=user_prompt,
        response_excerpt=partial_response or "",
    )
    if critique is None:
        # Persona file missing or Stage 2 fallback path. Already logged.
        return None

    _bump_session_count(session_id, n=1)

    # Mid-turn action_taken: `interrupt` if the verdict reached the veto
    # threshold; `log_only` otherwise (the post-loop ensemble may still
    # flag it). The actual cancel-and-inject dispatch happens in the
    # caller (messages.py); this field just records what we'd recommend.
    iv_dis = (CONFIG.get("inner_voice") or {}).get("disagreement") or {}
    veto_floor = float(iv_dis.get("veto_severity_threshold", 0.85))
    if (
        critique.disagrees
        and critique.severity >= veto_floor
        and critique.error is None
    ):
        critique.action_taken = "interrupt"
    else:
        critique.action_taken = "agreement" if not critique.disagrees else "log_only"

    try:
        _persist_critique(session_id=session_id, turn_id=turn_id, critique=critique)
    except Exception as e:
        logger.warning("inner_voice mid-turn critique persist failed: %s", e)

    if emit_sse is not None:
        try:
            await _emit_critique_sse(emit_sse, critique, turn_id=turn_id)
        except Exception as e:
            logger.warning("inner_voice mid-turn emit failed: %s", e)

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.aggregation_decision",
            {
                "ensemble_name": "mid_turn_drift",
                "personas_invoked": 1,
                "personas_disagreed": int(critique.disagrees),
                "severity_max": critique.severity,
                "severity_mean": critique.severity,
                "action_chosen": critique.action_taken,
                "thresholds": {
                    "veto_severity_threshold": veto_floor,
                },
                "rationale": (
                    f"mid-turn drift @ {stream_position_chars}c: "
                    f"sev={critique.severity:.2f} {'≥' if critique.severity >= veto_floor else '<'} {veto_floor}"
                ),
                "stage": "stage3_mid_turn",
                "stream_position_chars": stream_position_chars,
                "delta_index": delta_index,
            },
            turn_id=turn_id,
        )
    except Exception:
        pass

    try:
        if emit_sse is not None:
            await emit_sse(
                "inner_voice_observation_state",
                {"state": "observing", "stage": "mid_turn"},
            )
    except Exception:
        pass

    return critique


# ---------------------------------------------------------------------------
# Per-persona invocation — handles inference + raw-event logging only.
# Caller is responsible for SQLite persistence (after action_taken is set
# by aggregation).
# ---------------------------------------------------------------------------


async def _run_one_persona(
    *,
    session_id: str,
    turn_id: str,
    persona_name: str,
    user_prompt: str,
    response_excerpt: str,
) -> Critique | None:
    """Invoke one persona end-to-end: load prompt, call critic, log events.

    Does NOT write to SQLite — the caller persists after aggregation has
    set the final `action_taken`. Sets `event_log_offset` and
    `raw_response_offset` on the returned Critique so the caller can
    forward them to `record_inner_voice_critique`.
    """
    loaded = _read_persona_file(persona_name)
    if loaded is None:
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_skipped",
                {"reason": "persona_file_missing", "persona": persona_name},
                turn_id=turn_id,
            )
        except Exception:
            pass
        return None

    meta, body = loaded
    persona_version = meta.get("version") if isinstance(meta, dict) else None

    invoked_offset: int | None = None
    try:
        invoked_offset = _event_log.log_event(
            session_id,
            "inner_voice.persona_invoked",
            {
                "persona": persona_name,
                "persona_version": persona_version,
                "system_prompt": body,
                "user_prompt": user_prompt,
            },
            turn_id=turn_id,
        )
    except Exception as e:
        logger.warning("persona_invoked log failed: %s", e)

    started = time.monotonic()
    critique = await _critic.call_critic(
        persona=persona_name,
        persona_version=persona_version,
        persona_system_prompt=body,
        user_prompt=user_prompt,
        response_excerpt=response_excerpt,
    )
    wall_ms = int((time.monotonic() - started) * 1000)

    raw_offset: int | None = None
    try:
        raw_offset = _event_log.log_event(
            session_id,
            "inner_voice.persona_response_raw",
            {
                "persona": persona_name,
                "raw": critique.raw_response,
                "input_tokens": critique.input_tokens,
                "output_tokens": critique.output_tokens,
                "latency_ms": critique.latency_ms,
                "wall_ms": wall_ms,
                "error": critique.error,
                "parse_attempts": critique.parse_attempts,
            },
            turn_id=turn_id,
        )
    except Exception as e:
        logger.warning("persona_response_raw log failed: %s", e)

    if critique.error is None and critique.parse_attempts > 0:
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_response_parsed",
                {
                    "persona": persona_name,
                    "disagrees": critique.disagrees,
                    "severity": critique.severity,
                    "reason": critique.reason,
                    "suggested_action": critique.suggested_action,
                },
                turn_id=turn_id,
            )
        except Exception as e:
            logger.warning("persona_response_parsed log failed: %s", e)
    elif critique.error is not None:
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_parse_failure",
                {
                    "persona": persona_name,
                    "error": critique.error,
                    "parse_attempts": critique.parse_attempts,
                    "raw_excerpt": (critique.raw_response or "")[:500],
                },
                turn_id=turn_id,
            )
        except Exception as e:
            logger.warning("persona_parse_failure log failed: %s", e)

    # Stash the offsets on the Critique for the caller's _persist_critique.
    # Critique doesn't have these as dataclass fields (intentionally — they
    # belong to the row, not the verdict), so we set them as attribute-only.
    critique._event_log_offset = invoked_offset  # type: ignore[attr-defined]
    critique._raw_response_offset = raw_offset    # type: ignore[attr-defined]

    return critique


def _persist_critique(
    *,
    session_id: str,
    turn_id: str,
    critique: Critique,
) -> None:
    """Write one Critique to the `inner_voice_critiques` SQLite table.

    Reads forensic offsets from the side-channel attributes set by
    `_run_one_persona`. Defensive about their absence (set to None).
    """
    invoked_offset = getattr(critique, "_event_log_offset", None)
    raw_offset = getattr(critique, "_raw_response_offset", None)
    record_inner_voice_critique(
        session_id=session_id,
        turn_id=turn_id,
        persona=critique.persona,
        persona_version=critique.persona_version,
        model=critique.model,
        input_tokens=critique.input_tokens,
        output_tokens=critique.output_tokens,
        latency_ms=critique.latency_ms,
        disagrees=critique.disagrees,
        severity=critique.severity,
        reason=critique.reason,
        suggested_action=critique.suggested_action,
        action_taken=critique.action_taken,
        anchor_response_excerpt=critique.anchor_response_excerpt,
        event_log_offset=invoked_offset,
        raw_response_offset=raw_offset,
        prompt_hash=critique.prompt_hash,
        parse_attempts=critique.parse_attempts,
    )


# ---------------------------------------------------------------------------
# Aggregation — Stage 3 threshold logic
# ---------------------------------------------------------------------------


def _aggregate(critiques: list[Critique]) -> dict[str, Any]:
    """Return a summary dict over a list of Critiques.

    Stage 3 threshold logic:

      - **No critiques** (e.g. all personas errored) → `no_op` with a
        rationale that flags the gap.
      - **No disagreements** → `agreement`.
      - **Some disagreements**:
          - `severity_max >= veto_severity_threshold` → `veto_proposed`
            (end-of-turn dispatch is Stage 4; we record the verdict).
          - `severity_max >= severity_threshold` OR
            `disagree_count >= count_threshold` → `nudge_proposed`.
          - Otherwise → `log_only` (sub-threshold dissent — useful
            forensic data, no action).

    Stage 4 promotes `nudge_proposed` and `veto_proposed` into actual
    `steer` / `escalate` / `interrupt` dispatches. Stage 3 keeps them as
    proposals so the bench-comparison signal from Stage 2 baseline is
    apples-to-apples.
    """
    if not critiques:
        return {
            "disagree_count": 0,
            "severity_max": 0.0,
            "severity_mean": 0.0,
            "action_chosen": "no_op",
            "rationale": "ensemble produced no critiques (all personas errored or skipped)",
        }

    iv_dis = (CONFIG.get("inner_voice") or {}).get("disagreement") or {}
    nudge_floor = float(iv_dis.get("severity_threshold", 0.6))
    count_threshold = int(iv_dis.get("count_threshold", 2))
    veto_floor = float(iv_dis.get("veto_severity_threshold", 0.85))

    disagreeing = [c for c in critiques if c.disagrees and c.error is None]
    disagree_count = len(disagreeing)
    severity_max = max((c.severity for c in disagreeing), default=0.0)
    severity_mean = (
        sum(c.severity for c in disagreeing) / len(disagreeing) if disagreeing else 0.0
    )

    if disagree_count == 0:
        return {
            "disagree_count": 0,
            "severity_max": 0.0,
            "severity_mean": 0.0,
            "action_chosen": "agreement",
            "rationale": "all personas agreed (or only errored personas reported)",
        }

    if severity_max >= veto_floor:
        return {
            "disagree_count": disagree_count,
            "severity_max": severity_max,
            "severity_mean": severity_mean,
            "action_chosen": "veto_proposed",
            "rationale": (
                f"severity_max={severity_max:.2f} ≥ veto_severity_threshold={veto_floor:.2f}; "
                f"end-of-turn dispatch reserved for Stage 4 (currently log-only)"
            ),
        }

    if severity_max >= nudge_floor or disagree_count >= count_threshold:
        triggers: list[str] = []
        if severity_max >= nudge_floor:
            triggers.append(f"severity_max={severity_max:.2f} ≥ {nudge_floor:.2f}")
        if disagree_count >= count_threshold:
            triggers.append(f"disagree_count={disagree_count} ≥ {count_threshold}")
        return {
            "disagree_count": disagree_count,
            "severity_max": severity_max,
            "severity_mean": severity_mean,
            "action_chosen": "nudge_proposed",
            "rationale": (
                f"thresholds tripped: {' AND '.join(triggers)}; "
                f"end-of-turn dispatch reserved for Stage 4 (currently log-only)"
            ),
        }

    return {
        "disagree_count": disagree_count,
        "severity_max": severity_max,
        "severity_mean": severity_mean,
        "action_chosen": "log_only",
        "rationale": (
            f"sub-threshold dissent: severity_max={severity_max:.2f} < {nudge_floor:.2f} "
            f"AND disagree_count={disagree_count} < {count_threshold}"
        ),
    }


def _resolve_per_critique_action(critique: Critique, agg: dict[str, Any]) -> str:
    """Decide what `action_taken` value to record for one critique.

    Per-critique mapping rules:
      - `error is not None` → `log_only` (no useful verdict).
      - `disagrees == False` → `agreement` (the persona was on-board).
      - `disagrees == True` AND aggregate decided to dispatch → `log_only`
        (Stage 3 doesn't dispatch end-of-turn; Stage 4 promotes to `steer`/
        `escalate`).
      - Otherwise (`disagrees == True`, agg says no dispatch) → `log_only`.

    These all collapse to `agreement` or `log_only` at end-of-turn for
    Stage 3. The aggregation event captures the would-be action separately
    so the per-critique field can stay schema-stable.
    """
    if critique.error is not None:
        return "log_only"
    if not critique.disagrees:
        return "agreement"
    return "log_only"


# ---------------------------------------------------------------------------
# SSE emit shim
# ---------------------------------------------------------------------------


async def _emit_critique_sse(
    emit_sse: EmitFn,
    critique: Critique,
    *,
    turn_id: str,
) -> None:
    """Push the user-visible critique summary onto the active turn's events.

    Wire shape (matches `inner_voice_critique` on the frontend):
        {
          persona, severity, reason,
          disagrees, suggested_action, action_taken,
          anchor_turn_id, prompt_hash,
        }
    """
    await emit_sse(
        "inner_voice_critique",
        {
            "persona": critique.persona,
            "persona_version": critique.persona_version,
            "severity": critique.severity,
            "disagrees": critique.disagrees,
            "reason": critique.reason,
            "suggested_action": critique.suggested_action,
            "action_taken": critique.action_taken,
            "anchor_turn_id": turn_id,
            "prompt_hash": critique.prompt_hash,
            "error": critique.error,
        },
    )


# ---------------------------------------------------------------------------
# Config gates
# ---------------------------------------------------------------------------


def _is_enabled() -> bool:
    """Read `inner_voice.critic.enabled` (defaults true) — but only if the
    parent `inner_voice` block is present. The Inner-Voice tab session flag
    gates per-session; this gate is the kill-switch for emergency disable
    without a redeploy.
    """
    iv = CONFIG.get("inner_voice")
    if not iv:
        return False
    crit = iv.get("critic") or {}
    if "enabled" in crit:
        return bool(crit["enabled"])
    return True


def _is_mid_turn_drift_enabled() -> bool:
    """Read `inner_voice.mid_turn_drift.enabled` (defaults true).

    Independent of the post-loop critic gate so the two can be A/B'd
    separately. Cost note: each fire is one Brain 2 call, gated by
    `max_concurrent_personas` and `max_critiques_per_session` like the
    end-of-turn personas.
    """
    iv = CONFIG.get("inner_voice")
    if not iv:
        return False
    mtd = iv.get("mid_turn_drift") or {}
    if "enabled" in mtd:
        return bool(mtd["enabled"])
    return True


def get_mid_turn_drift_config() -> dict[str, Any]:
    """Public accessor for messages.py — returns the mid-turn drift config
    block with defaults applied. Caller uses these to decide when to fire.
    """
    iv = CONFIG.get("inner_voice") or {}
    mtd = dict(iv.get("mid_turn_drift") or {})
    mtd.setdefault("enabled", True)
    mtd.setdefault("min_chars_before_first_check", 250)
    mtd.setdefault("check_every_chars", 500)
    mtd.setdefault("max_checks_per_turn", 4)
    return mtd


# ---------------------------------------------------------------------------
# Mid-turn cancel-and-inject ambient builder
# ---------------------------------------------------------------------------


def make_drift_cancel_ambient(
    *,
    turn_id: str,
    persona: str,
    severity: float,
    reason: str,
    partial_excerpt: str,
) -> dict[str, str]:
    """Build the AmbientPrefetchEntry kwargs that surface a mid-turn cancel.

    The producer here is the Brain 2 verdict: drift_detector saw Brain 1
    confabulating partway through the turn and recommended `veto`. The
    prefetch entry describes (a) that the prior turn was cancelled, (b)
    why, and (c) what the agent should do on the next attempt.

    Same shape as `make_completion_nudge_entry` in heuristics.py — kwargs
    passable directly into `AmbientPrefetchEntry(**kwargs, enqueued_at=...)`.
    """
    excerpt = (partial_excerpt or "")[:300]
    return {
        "source": f"inner_voice:mid_turn_drift:{persona}",
        "summary": (
            f"Previous turn ({turn_id}) cancelled mid-stream by Inner Voice "
            f"({persona}, severity {severity:.2f}) — drift detected before "
            f"completion."
        ),
        "content": (
            "Inner Voice (#345 Stage 3) interrupted your last turn mid-stream "
            f"because {persona} flagged it at severity {severity:.2f}. "
            "The verdict's reason is below. Re-attempt the task — verify "
            "your claims with tool calls before asserting capabilities or "
            "facts you don't have evidence for. If the question genuinely "
            "has no answer in this codebase, say so directly.\n\n"
            f"Reason: {reason}\n\n"
            f"Cancelled output (first 300 chars): {excerpt!r}"
        ),
        "dedup_key": f"inner_voice:mid_turn_drift:{turn_id}",
    }


__all__ = [
    "run_post_loop_critique",
    "run_mid_turn_drift_check",
    "get_mid_turn_drift_config",
    "make_drift_cancel_ambient",
    "_build_user_prompt",
    "_load_recent_transcript",
    "_select_ensemble_for_turn",
    "_aggregate",
    "_top_matched_skills",
    "_format_matched_skills_block",
]
