"""Post-session capture orchestrator.

Runs in the background after a turn completes:
  1. Export the session as searchable markdown to the vault (immediate — for QMD index).
  2. Once per session: call the secondary model for a 2-4 sentence summary and
     append it to today's daily note.
  3. Every time the session gains ≥3 user messages since the last extraction:
     extract durable facts from the messages after the watermark.

Steps 2 and 3 have separate gates, which is the fix for #1159: they used to
share the `captured` boolean, the summary half won it on turn 1, and nothing
extracted a fact again. Step 1 and the summary run for every platform; step 3
runs only for sessions a human reads.

Also handles the focus-topic extraction invoked mid-session by prefetch.
Trivial sessions still get no facts — inline fact_add during conversation and
nightly extraction handle structured facts.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from app.paths import SESSIONS_DIR
from app.sessions_io import (is_conversation_session, is_user_session,
                             mutate_session)
from app.secondary_models import (
    _sync_secondary_capture_call,
    _sync_secondary_fact_extraction,
    _sync_secondary_focus_extraction,
)


logger = logging.getLogger("lloyd-server")

from app.paths import VAULT_BACKGROUND_SESSIONS_DIR, VAULT_SESSIONS_DIR


def _transcript_line(msg: dict) -> Optional[str]:
    """The transcript line for one message, or None if it is not transcript content.

    Dropped: any role other than `user`/`assistant` — which is what keeps a
    `thinking` row carrying a whole chain of thought out of every secondary-model
    prompt — and Claude Code's control blocks and envelope prefixes. Both
    transcript builders render through here so their bytes cannot drift;
    `eval/secondary_routing_eval.py` pins a hash over one of them.
    """
    role = msg.get("role", "")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content", "")
    if isinstance(content, list):
        text = "\n".join(
            t for t in (
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ) if t
        )
    elif isinstance(content, str):
        text = content
    else:
        return None
    if not text.strip():
        return None
    stripped = text.strip()
    if any(stripped.startswith(pfx) for pfx in (
        "<daily_notes>", "<memory>", "<context>", "<system-reminder>",
        "[cron:", "[System Message]", "[autonomy:",
    )):
        return None
    return f"{'USER' if role == 'user' else 'ASSISTANT'}: {text[:600]}"


def _build_capture_transcript(messages: list, max_chars: int = 4000) -> str:
    """Extract user/assistant text from messages, truncated to max_chars.

    Whole-session shape, spent head-and-tail: the first `max_chars/2` and last
    `max_chars/2` characters, so a session longer than one budget loses its
    middle and keeps its opening. That is the right shape for the capture
    summary — a 2-4 sentence recap wants the premise and the latest state — and
    it is the shape `eval/secondary_routing_eval.py` hashes, so its bytes are
    pinned here and by that file's own input hash.

    The fact extractor does not use it. A window that silently drops its own
    middle cannot say which messages it stands for, and advancing a watermark
    off a guess is how #1159's loss would come back under a new name; see
    `_build_fact_transcript`.
    """
    lines: list[str] = []
    for msg in messages:
        line = _transcript_line(msg)
        if line is not None:
            lines.append(line)

    result = "\n".join(lines)
    if len(result) > max_chars:
        half = max_chars // 2
        result = result[:half] + "\n[...truncated...]\n" + result[-half:]
    return result

#: Same budget the summary path spends, and deliberately the same number: it is
#: the size the secondary engine's prompt budget was tuned against, and this
#: shares the engine with the summary and the title.
FACT_TRANSCRIPT_BUDGET = 4000


def _build_fact_transcript(messages: list, *, start: int = 0,
                           max_chars: int = FACT_TRANSCRIPT_BUDGET) -> tuple[str, int]:
    """Transcript of the messages from `start` on, plus how far it reaches.

    Returns `(transcript, covered)` where every message at index `< covered` is
    inside the returned text — so a caller may safely advance a watermark to
    `covered` and mean it.

    Two properties `_build_capture_transcript` cannot offer, and which the
    summary path does not need because it has no watermark to advance. It spends
    its budget head-and-tail over the whole slice, so (a) the middle is invisible
    and (b) nothing in the returned string says which messages it stands for.
    Asking the extraction question from that shape is what #1159 is about, and
    guessing the covered count from a trimmed string is how a fix would end up
    advancing past messages nobody read: they would then never be extracted,
    which is the same loss with a new name.

    So it fills forward from `start` and stops at the last message that fits
    whole. A session longer than one budget drains across successive passes —
    the gate is "≥3 user messages past the watermark", and after a pass that
    covered part of a backlog there are still ≥3 unseen ones, so the next
    turn-end pass takes the next slice. One call per turn while a backlog exists
    is as fast as the single-slot engine allows anyway. The cost is honest and
    stated: the oldest unseen turns are extracted first, so the newest fact
    lands a pass or two later than it would with a tail-first fill.
    """
    chunks: list[str] = []
    used = 0
    covered = start
    for idx in range(max(0, start), len(messages)):
        msg = messages[idx]
        line = _transcript_line(msg)
        if line is None:
            # A `thinking` row, a blank block, or a control/envelope prefix: the
            # extractor has no use for it, and passing one over is not skipping
            # content, so `covered` still counts it.
            covered = idx + 1
            continue
        cost = len(line) + (1 if chunks else 0)
        if used + cost > max_chars:
            if not chunks:
                # Nothing rendered fit — the first rendered line is longer than
                # the whole budget. Reporting `covered` rather than `start` keeps
                # the rows already passed over (skipped `thinking` rows, control
                # prefixes) marked as seen: dropping them here would re-walk them
                # on every future pass without ever advancing.
                return "", covered
            return "\n".join(chunks), covered
        chunks.append(line)
        used += cost
        covered = idx + 1
    return "\n".join(chunks), covered


#: Session-file key holding the message index the last extraction pass covered.
#: Separate from `captured` on purpose: `captured` means "this session's summary
#: is in the daily note", which is a once-per-session thing; extraction is not.
#: The single boolean did double duty and the summary half won — see #1159.
FACT_WATERMARK_KEY = "fact_watermark"

#: New user messages required before another extraction call fires.
#:
#: Event-gated rather than per turn because the secondary engine is
#: single-tenant (`llama.cpp --parallel 1`, llm/CLAUDE.md): extraction shares
#: that queue with session titling and the capture summary, and a call per turn
#: would sit in front of the user's own work. Same reasoning that made titling
#: geometric at `app/session_titles.py:11-21`; the number 3 is the threshold the
#: old `user_msg_count >= 3` gate used, kept so the first pass after three user
#: messages behaves exactly as the old code did on its last pass.
FACT_EXTRACT_MIN_NEW_USER_MSGS = 3

#: Below this the slice is a greeting, not a conversation. Same floor the
#: summary path uses at `_post_session_capture`.
_MIN_FACT_TRANSCRIPT_CHARS = 50


def _fact_watermark(data: dict) -> int:
    """The message index the last extraction pass covered (0 if never).

    Defensively non-negative: the field is written by this module and read from
    a JSON file a person can edit, and a negative start would make
    `messages[start:]` hand the extractor the *tail* of the session while the
    watermark still claimed otherwise.

    And a watermark the file cannot vouch for reads as *never extracted*, which is
    the load-bearing half. The message list is shared mutable state that a whole
    other request path replaces wholesale: manual `/compact` assigns
    `data["messages"] = new_messages` under `mutate_session`
    (`app/routers/messages.py:1723-1742`) and never touches this key, so a 20- or
    30-message session compacted to 10 leaves a stamp of 20 or 30 counting against
    a list that no longer exists. Read raw on a 16-message file, `messages[20:]`
    is empty — forever, however many turns arrive. That is #1159's silence again,
    reinstated by the fix.

    Neither reading of a stale stamp is *true*, so the choice here is a stated
    loss, not a fact:

      - clamping to `len(messages)` claims the compacted tail was extracted. It
        may have been — but it claims that about the turns appended *after* the
        compaction too, which were never in any transcript, so the session stays
        silent until enough new turns push the list past the old number. Wrong in
        the direction this item exists to close.
      - reading 0 re-extracts the surviving turns, which spends the single
        engine slot on a window that was already answered. The store refuses a
        byte-identical re-add — `(entity, text_hash)` across categories, landed
        as #499, which is also what covers this file's direct write — but it
        cannot see a paraphrase, and the secondary model paraphrases, so the
        re-extraction returns near-duplicates the guard lets through.

    Duplicates win because they are the recoverable half: a dream pass or
    `fact_resolve` can expire a duplicate, and `forget` can expire it; a fact the
    window silently skipped leaves no trace that it was ever owed. The cost is
    bounded and self-limiting too — one re-extraction per wholesale replacement,
    since the next pass re-stamps the real coverage and the gate closes again.

    Clamping here rather than patching the compaction route is also the only
    place that covers every wholesale writer, including a restore over a session
    file, not just the one route someone happened to think of.
    """
    raw = data.get(FACT_WATERMARK_KEY)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    messages = data.get("messages")
    if isinstance(messages, list) and raw > len(messages):
        return 0
    return raw


#: Sessions with an extraction pass inside the model call, in this process.
#:
#: The watermark decides from state read *before* the call and applies after it,
#: so on its own it closes neither half of the #1159 hazard: two passes that read
#: the same watermark both spend a call on a single-slot engine, and then both
#: write. The store's `(entity, text_hash)` guard (#499) refuses the byte-identical
#: re-add, but a refusal is not an idempotent run: the engine call is already
#: spent by the time it happens, and the model's second answer is a paraphrase
#: the key cannot see. So the gate is this set — a pass that finds its session
#: already in it returns without calling.
#:
#: Process-local, which is the honest scope: every turn's capture pass is
#: dispatched by the backend process (`app/routers/messages.py:1309, :1395`), so
#: the passes that can interleave are in one process. The compare-before-write in
#: `_set_fact_watermark` is what covers a second writer, or a restart between the
#: call and the write, at the price of a redundant call rather than of a
#: duplicated fact or an unread stretch of conversation.
_in_flight: set[str] = set()


def _set_fact_watermark(covered: int, expect: int):
    """A `mutate_session` callback advancing the watermark, or refusing to.

    `covered` is the count of messages the transcript actually contained, so a
    turn appended during the 10+ second model call stays ahead of the watermark
    and the next pass picks it up instead of skipping it.

    `expect` is the watermark this pass read before it called the model. The
    write happens only if the file still says `expect`, and only if `covered` is
    inside the file it is being written into. Each refusal is a case where
    `covered` is not a true statement about that file — another pass in this
    process already advanced it, another process grew the list so `covered` was
    counted against a different one, or the file shaken under a roll or a restore
    — and in every one of them writing would advance the watermark past messages
    nobody extracted. A refusal costs a redundant call on the next pass; a wrong
    advance costs a stretch of conversation never extracted, which is the bug
    this module exists to fix.
    """
    def _apply(data: dict) -> None:
        current = _fact_watermark(data)
        # The compare-before-write alone is not enough: a caller whose `covered`
        # fell below the stored value would pass the compare (nothing raced it)
        # and rewind the window, which re-extracts the same turns: the store
        # refuses the byte-identical re-add (#499) but the engine slot is already
        # spent and the model's second answer is a paraphrase it cannot see. So
        # monotonic is checked on its own, and each refusal gets its own name.
        if covered < current:
            _apply.result = "lowered"
            return
        if current != expect:
            _apply.result = "already-advanced"
            return
        in_file = len(data.get("messages", []))
        if covered > in_file:
            _apply.result = "shrink"
            return
        data[FACT_WATERMARK_KEY] = covered
        _apply.result = "advanced"

    _apply.result = "not-run"
    return _apply


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


def _auto_captured_heading(now: datetime) -> str:
    """The `### Session HH:MM <ZONE> — Auto-captured` heading for one instant.

    The zone word is the instant's own `%Z`, not a typed literal:
    America/Los_Angeles prints `PDT` April–October and `PST` November–March,
    and the `PDT` hardcoded here previously mislabelled every winter capture
    for roughly four and a half months of the year (#601, umbrella #1189).
    The heading shape is byte-identical to the old one — `### Session ` +
    two-digit clock + space + three-letter zone + ` — Auto-captured` — which is
    the `^### Session \\d{2}:\\d{2}` … `Auto-captured` shape daily notes are
    grepped by; only the zone word's *source* changes, so summer headings are
    byte-for-byte what they were and winter headings stop lying. Takes the
    instant as an argument so a test can freeze a January one — with a live
    `datetime.now` the wrong-label bug was only observable in winter.
    """
    return f"### Session {now.strftime('%H:%M %Z')} — Auto-captured"


def _append_daily_note(session_id: str, summary: str,
                       now: datetime | None = None):
    """Append session summary to today's daily note (America/Los_Angeles).

    `now` is the wall-clock reading to record; default `datetime.now(pst)`. The
    parameter exists so a test can pin a winter instant — with a live
    `datetime.now` the wrong-label bug could only be observed in winter — and so
    the heading's time, its zone word, the note's LA-date filename and the
    fresh-file `timestamp:` frontmatter all come from ONE reading. They
    previously came from three separate `datetime.now` calls, which could
    disagree (a `timestamp:` whose date is not the filename's) at midnight.
    """
    from zoneinfo import ZoneInfo

    pst = ZoneInfo("America/Los_Angeles")
    if now is None:
        now = datetime.now(pst)
    today = now.strftime("%Y-%m-%d")
    daily_path = Path.home() / "obsidian" / "memory" / f"{today}.md"

    entry = f"\n---\n\n{_auto_captured_heading(now)}\n\n{summary}\n"

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
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
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

    # The same predicate the two listings apply. This used to be
    # `is_user_session` — the deny-list tuned for brief delivery — so a
    # transcript whose platform nobody had named was, by that list's own
    # design, "a user session", and landed in the corpus qmd embeds: the 3
    # automod `e2e-harness` smokes (93-203 messages of round transcripts each)
    # and the gate's `canary` turns are what a chat-shaped machine run looks
    # like, and a four-part run stamped `mission-control` was embedded too.
    # Being embedded is not the thing a delivery list should decide.
    root = (VAULT_SESSIONS_DIR if is_conversation_session(session_id, data)
            else VAULT_BACKGROUND_SESSIONS_DIR)
    out_dir = root / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = session_id.replace("/", "--")[:30]
    out_path = out_dir / f"{safe_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


async def _post_session_capture(session_id: str):
    """Background task: summarise a session and extract facts as it grows.

    Two halves with two independent gates, and the separation is the whole of
    #1159. The summary + daily note runs once per session and is guarded by
    `captured`. Fact extraction is guarded by `fact_watermark` and is *not*
    suppressed by `captured` — because one boolean could only mean one thing,
    and the summary half wrote it on turn 1 (turn 1 nearly always produces a
    non-trivial summary), so the old `user_msg_count >= 3` extraction branch
    below it was unreachable by construction. The retained server logs
    (`logs/server.err` … `server.err.10`, 2026-09-16 → 09-20) carry the string
    `facts extracted` exactly once, and its three surrounding lines put the
    export, the extraction and the summary in the same pass: the only session
    that ever got facts was one whose first capture pass did not arrive until it
    already held 4 user messages. No session has extracted on a second pass.

    Must never write a stale snapshot back to the session file — the
    secondary-model call can take 10+ seconds, during which new turns
    may append messages. Use `mutate_session` to apply the `captured`
    flag and the fact watermark atomically against current on-disk state.
    """
    try:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if not meta_path.exists():
            return

        # Snapshot is used only for read-only operations (markdown export,
        # transcript build). We never write this dict back.
        data = json.loads(meta_path.read_text())

        user_msgs = [
            m for m in data.get("messages", [])
            if m.get("role") == "user"
        ]
        if not user_msgs:
            return

        if not data.get("captured"):
            await _capture_summary_once(session_id, data)

        # Export and summary above run for every platform; writing facts into
        # the knowledge graph does not — see the platform note inside
        # `_capture_summary_once` for why a worker's notes to itself are not
        # things the user said.
        if is_user_session(data):
            await _extract_facts_past_watermark(session_id)

    except Exception as e:
        logger.warning(f"Post-session capture failed for {session_id}: {e}")


async def _capture_summary_once(session_id: str, data: dict):
    """Export the session, summarise it into today's daily note, latch `captured`.

    At most once per session — the caller skips this entirely when `captured` is
    set — and it writes that latch itself on every path that consumed the
    summary: a non-user platform, a `TRIVIAL` verdict, or a summary appended.
    The one path that does not latch is the sub-50-character transcript, where
    there was nothing to summarise *yet*, so a later pass is still free to try.

    Owns its exception handler so a failing summary cannot also eat the fact
    pass that runs after it.
    """
    try:
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

        await mutate_session(session_id, lambda d: d.__setitem__("captured", True))

        logger.info(f"Post-session capture: {session_id} — summary written to daily note")

    except Exception as e:
        logger.warning(f"Post-session capture failed for {session_id}: {e}")


async def _extract_facts_past_watermark(session_id: str) -> int:
    """Extract durable facts from the messages this session has not shown us yet.

    Returns the number of facts written (also the signal a test asserts on: zero
    return value with a non-zero call count means the model found nothing durable,
    which is a different outcome from the gate declining to ask it).

    Re-reads the session file rather than trusting the caller's snapshot: the
    pass that reached here may have spent 10+ seconds in the summary call, and
    the turns appended since are precisely the ones #1159 is about — a spoken
    conversation keeps arriving over `/api/voice/inject` while the first pass is
    still in flight.

    The gate is `>= FACT_EXTRACT_MIN_NEW_USER_MSGS` user messages *past the
    watermark*, so the call is event-gated and not per turn: the secondary
    engine is single-tenant, and a pass with nothing new must issue zero calls.
    Re-running this with no new messages therefore writes nothing.

    The watermark advances on any completed attempt, including one that returned
    no facts, so the same messages are never sent twice. The store refuses a
    byte-identical re-add (#499's `(entity, text_hash)` key) but is blind to a
    paraphrase, and this prompt asks the model to restate, so idempotence lives
    here rather than at the store. A raised call does not advance it: a transient engine
    refusal gets retried by the next pass instead of being consumed.
    """
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return 0

    data = json.loads(meta_path.read_text())
    messages = data.get("messages", [])
    watermark = _fact_watermark(data)
    new_user = [m for m in messages[watermark:] if m.get("role") == "user"]
    if len(new_user) < FACT_EXTRACT_MIN_NEW_USER_MSGS:
        return 0

    # The budget covers the unseen part, filled forward from the watermark
    # rather than head-and-tail, so `covered` can be the number of messages the
    # model actually read — see `_build_fact_transcript`. The count comes off the
    # returned slice and never off `len(messages)`: a turn appended during the
    # model call below has to stay ahead of the watermark for the next pass to
    # catch it, and a slice trimmed for length must not be reported as full
    # coverage, or the trimmed middle is lost with the file stamped as read.
    transcript, covered = _build_fact_transcript(messages, start=watermark)
    if covered <= watermark:
        # Not one rendered line fit inside the budget, which given the 600-char
        # per-line cap means the unseen rows are all `thinking` rows and control
        # prefixes. Advancing past them here would be right in spirit but is
        # already handled below by the too-short transcript, which does advance;
        # this branch is only the case where the builder refused at the first
        # line, and it must not spend the single engine slot on an empty prompt.
        logger.info(
            f"Post-session capture: {session_id} — nothing extractable in "
            f"messages {watermark}..{len(messages)} within the "
            f"{FACT_TRANSCRIPT_BUDGET}-char budget"
        )
        return 0

    # Claim the session with no `await` between the test and the add, so two
    # dispatches for one session cannot both decide they may call: the event loop
    # runs this much without yielding. The `finally` below is the only release,
    # which is why nothing between the add and the `try` returns early.
    if session_id in _in_flight:
        logger.info(
            f"Post-session capture: {session_id} — an extraction pass is already "
            f"running; this pass leaves the window to it"
        )
        return 0
    _in_flight.add(session_id)

    try:
        if len(transcript.strip()) < _MIN_FACT_TRANSCRIPT_CHARS:
            # Seen, and nothing in it. Advance so the same three-word turn does
            # not reopen the window on every future pass.
            apply = _set_fact_watermark(covered, expect=watermark)
            await mutate_session(session_id, apply)
            return 0

        facts = await asyncio.get_event_loop().run_in_executor(
            None, _sync_secondary_fact_extraction, transcript
        )
    except Exception as fe:
        logger.warning(f"Post-session fact extraction failed for {session_id}: {fe}")
        return 0
    finally:
        # If the call raised, the watermark stays where it was and the next pass
        # retries the same window: a refused call is not a consumed window, which
        # is the difference between a retry and a silently dropped stretch of
        # conversation. Same for a cancellation that lands mid-await.
        _in_flight.discard(session_id)

    if facts:
        _write_extracted_facts(facts, session_id)
        logger.info(
            f"Post-session capture: {session_id} — {len(facts)} facts extracted "
            f"(messages {watermark}..{covered})"
        )
    else:
        logger.info(
            f"Post-session capture: {session_id} — no durable facts in "
            f"{len(new_user)} new user messages past message {watermark}"
        )

    apply = _set_fact_watermark(covered, expect=watermark)
    await mutate_session(session_id, apply)
    if apply.result != "advanced":
        logger.info(
            f"Post-session capture: {session_id} — watermark write refused "
            f"({apply.result}): was {watermark}, target {covered}"
        )
    return len(facts or [])
