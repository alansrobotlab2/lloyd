"""Post-session capture orchestrator.

Runs in the background after a turn completes:
  1. Export the session as searchable markdown to the vault (immediate — for QMD index).
  2. Call the secondary model for a 2-4 sentence summary; append to today's daily note.
  3. If the session is substantive (≥3 user turns), extract durable facts.

Also handles the focus-topic extraction invoked mid-session by prefetch.
Facts are NOT written here for trivial sessions — inline fact_add during
conversation and nightly extraction handle structured facts.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from app.paths import SESSIONS_DIR
from app.sessions_io import is_user_session, mutate_session
from app.secondary_models import (
    _sync_secondary_capture_call,
    _sync_secondary_fact_extraction,
    _sync_secondary_focus_extraction,
)


logger = logging.getLogger("lloyd-server")

from app.paths import VAULT_BACKGROUND_SESSIONS_DIR, VAULT_SESSIONS_DIR


def _build_capture_transcript(messages: list, max_chars: int = 4000) -> str:
    """Extract user/assistant text from messages, truncate to max_chars."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text = "\n".join(t for t in text_parts if t)
        elif isinstance(content, str):
            text = content
        else:
            continue
        if not text.strip():
            continue
        stripped = text.strip()
        if any(stripped.startswith(p) for p in (
            "<daily_notes>", "<memory>", "<context>", "<system-reminder>",
            "[cron:", "[System Message]", "[autonomy:",
        )):
            continue
        label = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"{label}: {text[:600]}")

    result = "\n".join(lines)
    if len(result) > max_chars:
        half = max_chars // 2
        result = result[:half] + "\n[...truncated...]\n" + result[-half:]
    return result


def _write_extracted_facts(facts: list[dict], session_id: str):
    """Write extracted facts to the fact store via direct file append."""
    from agent_mcp.facts import _fact_add

    for f in facts:
        try:
            _fact_add({
                "entity": f["entity"],
                "category": "session-extracted",
                "fact": f["fact"],
                "confidence": 0.75,
                "provenance": "EXTRACTED",
                "source_doc": f"sessions/{session_id}",
            })
        except Exception as e:
            logger.warning(f"Failed to write fact '{f['fact'][:40]}...': {e}")


def _append_daily_note(session_id: str, summary: str):
    """Append session summary to today's daily note (PST)."""
    from zoneinfo import ZoneInfo

    pst = ZoneInfo("America/Los_Angeles")
    today = datetime.now(pst).strftime("%Y-%m-%d")
    now_time = datetime.now(pst).strftime("%H:%M")
    daily_path = Path.home() / "obsidian" / "memory" / f"{today}.md"

    entry = f"\n---\n\n### Session {now_time} PDT — Auto-captured\n\n{summary}\n"

    if not daily_path.exists():
        # OKF requires a non-empty `type` (scripts/vault/validate_okf.py), and
        # every pre-existing daily note uses this shape — see memory/2026-06-14.md.
        # The old header was `segment: agents` with no `type`, which was wrong
        # twice: the file lives under memory/, and each new calendar day was
        # born a conformance violation (item #519). Dumped rather than spelled
        # so the frontmatter is strict-parseable by construction, with the same
        # kwargs scripts/vault/okf_migrate.py uses to repair the older ones.
        frontmatter = yaml.safe_dump(
            {
                "segment": "memory",
                "tags": ["memory", "daily-notes"],
                "type": "note",
                "timestamp": datetime.now(pst).strftime("%Y-%m-%dT%H:%M:%S"),
            },
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).rstrip()
        daily_path.write_text(
            f"---\n{frontmatter}\n---\n\n# {today} Daily Notes\n\n## Sessions\n{entry}"
        )
    else:
        with open(daily_path, "a") as f:
            f.write(entry)


async def _maybe_extract_focus(session_id: str):
    """Background: extract conversation topics via secondary model for focus tracking."""
    try:
        from prefetch import _get_session_focus, FOCUS_EXTRACT_INTERVAL  # noqa: F401

        focus = _get_session_focus(session_id)
        if not focus or not focus.needs_topic_extraction():
            return

        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if not meta_path.exists():
            return
        data = json.loads(meta_path.read_text())
        messages = data.get("messages", [])

        recent = [m for m in messages[-10:] if m.get("role") in ("user", "assistant")]
        if len(recent) < 3:
            return

        lines = []
        for m in recent:
            content = m.get("content", "")
            if isinstance(content, list):
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                text = " ".join(t for t in text_parts if t)
            elif isinstance(content, str):
                text = content
            else:
                continue
            stripped = text.strip()
            if any(stripped.startswith(p) for p in ("<context>", "<system-reminder>", "<memory>", "<daily_notes>")):
                continue
            role = "USER" if m.get("role") == "user" else "ASSISTANT"
            lines.append(f"{role}: {text[:200]}")

        transcript = "\n".join(lines)
        if len(transcript) < 50:
            return

        topics = await asyncio.get_event_loop().run_in_executor(
            None, _sync_secondary_focus_extraction, transcript
        )

        # Record the attempt whether or not it produced topics. Before this,
        # an empty/failed extraction left `topics_turn` untouched, so
        # `needs_topic_extraction()` stayed true and the model call re-fired
        # on every subsequent turn instead of every FOCUS_EXTRACT_INTERVAL.
        focus.mark_topic_attempt()
        if topics:
            focus.set_topics(topics)
            logger.info(f"Focus extraction for {session_id}: {topics}")
        else:
            logger.info(f"Focus extraction for {session_id}: no topics returned")

    except Exception as e:
        logger.debug(f"Focus extraction failed for {session_id}: {e}")


def _export_session_markdown(session_id: str, data: dict) -> Optional[Path]:
    """Export a Lloyd session as searchable markdown to the vault sessions collection.

    Writes immediately (no LLM call) so QMD can index it within seconds.
    Format matches the old Hermes extract-session-log.py output for consistency.
    Returns the path written, or None on failure.

    **A background run exports to a different directory**, and the reason is
    `agent-services/scripts/qmd-watcher.sh`: it indexes and *embeds*
    `_pipeline/vault-derived/sessions/` on every change. The ~70
    session-backed worker transcripts a day that reach this function (479 in
    the week to 2026-09-10, against 99 chats) would each be an embedding job
    over the machine talking to itself, drowning the corpus that exists to
    answer questions about what the user and Lloyd discussed. `sessions-background/`
    sits outside the watch: still exported, still greppable, not embedded.

    Which background runs reach here is worth stating, because it is not all
    of them. `_post_session_capture` is fired by the chat path only, so the
    background sessions that arrive are the session-backed workers — autocode,
    autotriage, deep-research, youtube-digest — which are also the ones that
    rewrite this repo. An `autonomy` run and a `run_prompt_on_primary` job
    call `run_query` directly and never come through; their record is the
    session JSON `app/run_recorder.py` writes, which the Background tab reads
    and `grep` can read too. Deliberate, not an oversight: a second copy in
    the vault would buy a marginally nicer grep target and cost a vault write
    per run, on a path whose whole design rule is that recording must never
    be able to break the run.
    """
    from zoneinfo import ZoneInfo
    pst = ZoneInfo("America/Los_Angeles")

    created_at = data.get("created_at", "")
    try:
        dt = datetime.fromisoformat(created_at)
    except Exception:
        dt = datetime.now()
    date_str = dt.astimezone(pst).strftime("%Y-%m-%d")
    ts_str = dt.isoformat()

    lines: list[str] = []
    lines.append(f"# {session_id}")
    lines.append(f"# {ts_str}")
    model = data.get("model", "")
    if model:
        lines.append(f"# model: {model}")
    lines.append("")

    for msg in data.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text = "\n".join(t for t in text_parts if t)

            tool_uses = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
        elif isinstance(content, str):
            text = content
            tool_uses = []
        else:
            continue

        if role == "user":
            stripped = text.strip()
            if any(stripped.startswith(p) for p in (
                "<context>", "<system-reminder>", "<memory>", "<daily_notes>",
                "[cron:", "[System Message]", "[autonomy:",
            )):
                continue
            if not stripped or len(stripped) < 2:
                continue
            display = stripped[:600] if len(stripped) > 600 else stripped
            lines.append(f"user: {display}")

        elif role == "assistant":
            for tu in tool_uses:
                name = tu.get("name", "?")
                args = tu.get("input", {})
                arg_parts = []
                for k, v in (args.items() if isinstance(args, dict) else []):
                    if isinstance(v, str):
                        arg_parts.append(f"{k}={v[:200]}")
                    elif isinstance(v, (bool, int, float)):
                        arg_parts.append(f"{k}={v}")
                    else:
                        arg_parts.append(f"{k}=...")
                lines.append(f"tool_call: {name}({', '.join(arg_parts)})")

            if text.strip() and len(text.strip()) > 10:
                display = text.strip()[:500]
                lines.append(f"lloyd: {display}")

        elif role == "tool":
            result_text = text.strip()[:300] if text else "(empty)"
            is_error = msg.get("is_error", False)
            status = "ERROR" if is_error else "OK"
            lines.append(f"  → [{status}] {result_text}")

    if len(lines) <= 3:
        return None

    root = (VAULT_SESSIONS_DIR if is_user_session(data)
            else VAULT_BACKGROUND_SESSIONS_DIR)
    out_dir = root / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = session_id.replace("/", "--")[:30]
    out_path = out_dir / f"{safe_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


async def _post_session_capture(session_id: str):
    """Background task: extract summary from completed session via secondary model.

    Must never write a stale snapshot back to the session file — the
    secondary-model call can take 10+ seconds, during which new turns
    may append messages. Use `mutate_session` to apply the `captured`
    flag atomically against current on-disk state.
    """
    try:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if not meta_path.exists():
            return

        # Snapshot is used only for read-only operations (markdown export,
        # transcript build). We never write this dict back.
        data = json.loads(meta_path.read_text())

        if data.get("captured"):
            return

        user_msgs = [
            m for m in data.get("messages", [])
            if m.get("role") == "user"
        ]
        if not user_msgs:
            return

        try:
            md_path = _export_session_markdown(session_id, data)
            if md_path:
                logger.info(f"Post-session capture: {session_id} — markdown exported to {md_path}")
        except Exception as me:
            logger.warning(f"Session markdown export failed for {session_id}: {me}")

        # The markdown export happens for every platform — a background run's
        # transcript is worth having on disk, which is the whole point of
        # recording it. Everything below is not: the summary is a secondary
        # model call, the daily note is the user's own record of their day,
        # and fact extraction writes into the knowledge graph as if the
        # machine's notes to itself were things the user said. `autonomy` was
        # excluded from all of it from the start; `worker` never was, and
        # worker turns arrive through the chat path.
        if not is_user_session(data):
            await mutate_session(session_id, lambda d: d.__setitem__("captured", True))
            return

        transcript = _build_capture_transcript(data.get("messages", []))
        if len(transcript.strip()) < 50:
            return

        summary = await asyncio.get_event_loop().run_in_executor(
            None, _sync_secondary_capture_call, transcript
        )

        if not summary or summary.strip().upper() == "TRIVIAL":
            logger.info(f"Post-session capture: {session_id} — trivial, skipped")
            await mutate_session(session_id, lambda d: d.__setitem__("captured", True))
            return

        _append_daily_note(session_id, summary)

        user_msg_count = len([m for m in data.get("messages", []) if m.get("role") == "user"])
        if user_msg_count >= 3:
            try:
                facts = await asyncio.get_event_loop().run_in_executor(
                    None, _sync_secondary_fact_extraction, transcript
                )
                if facts:
                    _write_extracted_facts(facts, session_id)
                    logger.info(f"Post-session capture: {session_id} — {len(facts)} facts extracted")
            except Exception as fe:
                logger.warning(f"Post-session fact extraction failed for {session_id}: {fe}")

        await mutate_session(session_id, lambda d: d.__setitem__("captured", True))

        logger.info(f"Post-session capture: {session_id} — summary written to daily note")

    except Exception as e:
        logger.warning(f"Post-session capture failed for {session_id}: {e}")
