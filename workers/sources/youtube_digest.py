"""youtube-digest — one tracked-channel video per session: transcript → note → verdict.

Alan follows two YouTube channels to keep up with agent and model
techniques — AI Engineer (@aiDotEngineer) and Discover AI (@code4AI). Until
2026-09-08 the first was digested by autonomy task #75, which ran
`scripts/ai-engineer-monitor.py --process-one`: a script that fetched the
transcript, POSTed 12k characters of it to the primary with thinking off, and
wrote the note. Nothing of that was a transcript anyone could read. The
autonomy runner calls `run_query` directly, so there was no session and no
Inner Voice, and the only artifact was the note itself.

This source is the hybrid Alan asked for. The deterministic half stays in
`scripts/youtube_channel_monitor.py`: it knows the channel, owns `seen.json`,
lists new uploads, and fetches the transcript, metadata and link enrichment
into one bundle directory per video. The judgement half runs here as a **real
session** through `run_prompt_in_session`, Inner Voice on: the model reads the
whole transcript, writes the vault note, evaluates the video against
`eval/lloyd_profile.md` — would anything here improve Lloyd, and where would
it plug in — and files a draft backlog item when the answer is yes. Every
step is in the session list and the Inner Voice tab.

Three properties worth knowing before changing anything here:

* **The script owns the retry, not the queue.** The pool records an in-band
  `failed` and completes the item; only a raised exception is retried. So a
  failed session is reported to the script with `--fail`, which counts the
  attempt in `seen.json`, and `_is_retry_eligible` decides when `--pending`
  offers the video again. A `DrainActive` (a landing in progress) is not the
  video's fault: the entry stays `fetched` and is offered on the next tick.
* **Disk decides what happened.** The note must exist at the path the source
  chose and carry this video's id, or the turn failed however finished its
  text reads. A `FILED: #n` claim is checked against the backlog directory
  before it is recorded; an unverifiable id is kept as `filed_unverified`.
* **The turn reads untrusted text** (a transcript is whatever the uploader
  captioned), so it runs with a deny list: no shell, no code edits, no
  subagents, nothing that seeds queues or touches the self-modification
  loop. `Read`/`Write` stay because the note is the job; the vault is a git
  repo, and writes outside the expected prefixes are reported.

Open-source frameworks and tools may be proposed for direct adoption or
evaluation. Commercial products may not — the eval names the aspects worth
recreating locally instead. That rule is Alan's, and it is in the prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.paths import LLOYD_HOME, VAULT_ROOT
from workers.queue import WorkQueue, QueueItem
from workers.sources._common import (
    WORKER_SELFMOD_BAN, DrainActive, TurnTimeout, run_prompt_in_session,
)

logger = logging.getLogger("lloyd-workers.youtube-digest")

NAME = "youtube-digest"
#: Between backlog triage (55) and the research stream (70). A digest is
#: routine; it should not starve a selfmod round, and it should run before
#: the next research topic because Alan reads these.
DEFAULT_PRIORITY = 60

SCRIPT = LLOYD_HOME / "scripts" / "youtube_channel_monitor.py"
PROFILE_PATH = LLOYD_HOME / "eval" / "lloyd_profile.md"
BACKLOG_DIR = VAULT_ROOT / "backlog"
DEFAULT_CHANNELS: tuple[str, ...] = ("ai-engineer", "discover-ai")
EVAL_TAG = "youtube-eval"

#: What the script prints ahead of its machine-readable last line.
JSON_MARK = "@@JSON@@ "

#: A note has to be more than a heading to count as written.
_MIN_NOTE_BYTES = 400

#: Vault prefixes this turn is expected to write under. Anything else that
#: changes during the turn is reported in the run record.
EXPECTED_VAULT_PREFIXES = ("knowledge/", "backlog/", "projects/lloyd/channel-eval/")

#: What a digest turn may not touch. Same reasoning as `deep_research`:
#: the transcript is untrusted text. Unlike that source this one keeps
#: `Read` and `Write` (the bundle is read from disk and the note written
#: to a path the source chose) and `backlog_write_task` (filing is the job).
DISALLOWED: tuple[str, ...] = (
    *WORKER_SELFMOD_BAN,
    "Bash", "Edit", "Task",
    "http_request",
    "browser_evaluate", "browser_fill", "browser_type", "browser_click",
    "browser_press", "browser_cookies", "browser_drag", "browser_select",
    "autonomy_write_task", "autonomy_delete_task", "autonomy_run_task",
    "autonomy_config",
    "research_propose", "research_next", "research_complete",
)

VERDICTS = ("actionable", "worth_a_look", "background", "not_relevant")
SOURCE_KINDS = ("open-source", "commercial", "paper", "concept")
APPROACHES = ("adopt", "recreate", "experiment", "read", "none")
AREAS = ("model", "inference", "harness", "tools", "memory", "knowledge-graph",
         "retrieval", "skills", "autonomy", "selfmod", "eval", "voice", "research", "ui")
_RESULTS = ("written", "kept", "failed")
_FIELD_RE = re.compile(
    r"^(RESULT|NOTE|RELEVANCE|VERDICT|AREAS|SOURCE_KIND|APPROACH|IDEA|DUPLICATE_OF|FILED):\s*(.*)$",
    re.I)
_ID_RE = re.compile(r"#?\s*(\d+)")


class ScriptError(RuntimeError):
    """The monitor script did not produce its JSON line."""


# ── The prompt ───────────────────────────────────────────────────────────────

PROMPT = """\
[SYSTEM: You are running the "youtube-digest" worker job. Work autonomously \
and do not ask for confirmation.]

You are digesting one YouTube video for Lloyd's vault and judging whether it \
holds anything that would improve Lloyd. Everything you need is already on \
disk. Do not search for the video, do not fetch it, and do not look for other \
videos.

<video>
channel: {channel_name} (@{channel_handle})
title: {title}
video_id: {video_id}
url: {url}
published: {published}
transcript: {transcript_path} ({transcript_words} words, {transcript_lines} lines)
metadata: {meta_path}
existing_note: {existing_note}
enrichment_notes: {enrichment}
</video>

Work in this order:

1. **Read the transcript in full** with Read. It is wrapped at 100 characters \
so it pages normally; use offset/limit if one call does not return all of it. \
Read {meta_path} too — the description often carries the links the speaker \
only alludes to.

2. **Write the vault note** to exactly this path, with Write:
   {target_note}
{existing_note_instruction}
   Use exactly this shape — front matter keys verbatim, values filled in:

---
segment: knowledge
tags: [youtube, ai, <three to six specific topic tags>]
type: video-note
domain: ai
source: {url}
video_id: {video_id}
channel: {channel_handle}
published: {published}
---

# {title}

## Executive Summary
Two to four dense sentences: what it covers, the core argument, why it matters.

## Key Points
Four to eight bullets, each a specific claim or finding, with the numbers the video gives.

## Technical Details
The meat: architectures, methods, algorithms, measured results, the exact terms the speaker uses. Sub-headings as needed.

## Tools & Frameworks Mentioned
- name: how it was discussed, and whether it is open source, a paper, or a product

## Related Resources
### GitHub
### Papers
### Links
Real URLs from the transcript or description; [[wiki links]] to the enrichment notes listed above; "None detected" when empty.

## Open Questions
Unresolved questions the video raises.

   Write from the transcript, not from what you already know about the topic. \
No preamble in the file and no code fence around it.

3. **Evaluate it for Lloyd.** Read {profile_path} — it describes Lloyd's \
stack, what already exists, and the standing problems. Decide whether this \
video contains a specific technique, tool, framework, model or finding that \
could concretely improve Lloyd, and where it would plug in.
   - `actionable` (relevance 70-100): a specific change with a measurable \
acceptance, feasible on two 24 GB GPUs with no cloud dependency.
   - `worth_a_look` (40-69): promising, but needs reading before a change can \
be named.
   - `background` (10-39): useful context, nothing to do.
   - `not_relevant` (0-9): off-topic for Lloyd.
   Rules: an **open-source** framework, tool or model may be proposed for \
direct adoption or evaluation — name the repo. A **commercial** product is \
never adopted; name the specific aspects worth recreating locally, if any. A \
**paper** or concept becomes a bounded experiment against an existing metric. \
Restatements of what Lloyd already does, generic advice and hype are \
`background` at most. Be skeptical: most videos are `background`.
   Already tracked by this eval — compare before filing, and if the idea is \
the same do not file, set DUPLICATE_OF instead:
{tracked}

4. **If the verdict is `actionable` and it is not a duplicate, file it** with \
`backlog_write_task`: board `lloyd`, no `task_id`, status `draft`, tags \
`{eval_tag}`, `{channel_key}`, plus the area tags. Name: a short imperative \
title. Description, written as a handoff a fresh session can act on alone:
   - **Source:** {channel_name} — "{title}" — {url} (published {published}); \
vault note [[{note_stem}]]
   - **Area / source kind / approach / effort**
   - **What it is** — the technique or tool, one paragraph
   - **Why it could improve Lloyd** — where it plugs in, which standing problem it hits
   - **Evidence from the video** — the claims and numbers that support it
   - **Proposed evaluation** — numbered, bounded steps
   - **Acceptance** — the measurement that would show it worked
   - **Risks and open questions**
   The tool returns the new id. File at most one item for this video.

5. **End your final message with exactly this block and nothing after it:**

RESULT: <written|kept|failed>
NOTE: {target_note}
RELEVANCE: <0-100>
VERDICT: <actionable|worth_a_look|background|not_relevant>
AREAS: <comma-separated, from: {areas}>
SOURCE_KIND: <open-source|commercial|paper|concept>
APPROACH: <adopt|recreate|experiment|read|none>
IDEA: <one line naming the specific thing, or none>
DUPLICATE_OF: <#id or none>
FILED: <#id or none>

`written` means you wrote the note at the path above; `kept` means an \
existing note was good enough and you left it; `failed` means you could not \
produce a note — say why in one line before the block.
"""

_EXISTING = (
    "   A note for this video already exists at that path. Read it first. "
    "Rewrite it in place only if the full transcript lets you make it clearly "
    "better; otherwise leave it and answer `RESULT: kept`. The eval in step 3 "
    "runs either way."
)
_FRESH = "   No note exists for this video yet."


def build_prompt(meta: dict, tracked: list[dict]) -> str:
    """Render the digest prompt for one bundle.

    `tracked` is the list of backlog items this eval has already filed, so a
    channel that returns to a theme every week (Discover AI and "the harness",
    say) does not file it every week.
    """
    enrichment = meta.get("enrichment") or {}
    notes = []
    for r in list(enrichment.get("github") or []) + list(enrichment.get("papers") or []):
        p = (r or {}).get("note_path")
        if p:
            notes.append(Path(p).stem)
    tracked_lines = [f"   - #{t['id']} {t['title']}" for t in tracked] or ["   - none yet"]
    target = str(meta.get("target_note") or "")
    return PROMPT.format(
        channel_name=meta.get("channel_name", ""),
        channel_handle=meta.get("channel_handle", ""),
        channel_key=meta.get("channel_key", ""),
        title=meta.get("title", ""),
        video_id=meta.get("video_id", ""),
        url=meta.get("url", ""),
        published=meta.get("published", "") or "unknown",
        transcript_path=meta.get("transcript_path", ""),
        transcript_words=meta.get("transcript_words", "?"),
        transcript_lines=meta.get("transcript_lines", "?"),
        meta_path=meta.get("meta_path", ""),
        existing_note=meta.get("existing_note") or "none",
        enrichment=", ".join(f"[[{n}]]" for n in notes) or "none",
        target_note=target,
        note_stem=Path(target).stem if target else "",
        existing_note_instruction=_EXISTING if meta.get("existing_note") else _FRESH,
        profile_path=str(PROFILE_PATH),
        tracked="\n".join(tracked_lines),
        eval_tag=EVAL_TAG,
        areas=", ".join(AREAS),
    )


# ── The RESULT block ─────────────────────────────────────────────────────────


def _int_or_none(text: str) -> Optional[int]:
    m = re.search(r"\d+", text or "")
    return int(m.group(0)) if m else None


def _id_or_none(text: str) -> Optional[int]:
    text = (text or "").strip()
    if not text or text.lower().startswith("none"):
        return None
    m = _ID_RE.search(text)
    return int(m.group(1)) if m else None


def parse_result(text: str) -> Optional[dict]:
    """Pull the trailing RESULT block out of the turn's final text.

    Parsed from the LAST `RESULT:` onward, as `deep_research.parse_result`
    does: a model that states an outcome, reconsiders and restates would
    otherwise have its first verdict paired with its last evidence. Fields
    outside their vocabulary come back as None rather than failing the parse —
    the note is the primary product and is verified on disk separately.
    """
    lines = (text or "")[-6000:].splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("RESULT:"):
            start = i
    if start is None:
        return None

    fields: dict[str, list[str]] = {}
    current = None
    for line in lines[start:]:
        m = _FIELD_RE.match(line.strip())
        if m:
            current = m.group(1).upper()
            fields[current] = [m.group(2)]
        elif current:
            fields[current].append(line)

    def one(key: str) -> str:
        return " ".join(" ".join(fields.get(key, [])).split()).strip().strip("`'\"")

    result = one("RESULT").lower().split()
    result = result[0] if result else ""
    verdict = one("VERDICT").lower().replace(" ", "_").replace("-", "_")
    source_kind = one("SOURCE_KIND").lower().replace("_", "-").replace(" ", "-")
    approach = one("APPROACH").lower()
    relevance = _int_or_none(one("RELEVANCE"))
    if relevance is not None:
        relevance = max(0, min(100, relevance))
    areas = [a.strip().lower().replace("_", "-").replace(" ", "-")
             for a in re.split(r"[,;/]", one("AREAS")) if a.strip()]
    idea = one("IDEA")
    return {
        "result": result if result in _RESULTS else None,
        "note": one("NOTE").strip("() "),
        "relevance": relevance,
        "verdict": verdict if verdict in VERDICTS else None,
        "areas": [a for a in areas if a in AREAS],
        "source_kind": source_kind if source_kind in SOURCE_KINDS else None,
        "approach": approach if approach in APPROACHES else None,
        "idea": "" if idea.lower() == "none" else idea,
        "duplicate_of": _id_or_none(one("DUPLICATE_OF")),
        "filed": _id_or_none(one("FILED")),
    }


# ── Disk checks ──────────────────────────────────────────────────────────────


def _note_is_real(path: Path, video_id: str) -> bool:
    """The note the source asked for, and not some other file at that path."""
    try:
        if not path.is_file():
            return False
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) < _MIN_NOTE_BYTES or not data.startswith(b"---"):
        return False
    head = data[:4000].decode("utf-8", "replace")
    return bool(re.search(rf"^video_id:\s*{re.escape(video_id)}\s*$", head, re.M))


def _read_head(path: Path, nbytes: int = 6000) -> str:
    try:
        with path.open("r", errors="ignore") as f:
            return f.read(nbytes)
    except OSError:
        return ""


def tracked_items(backlog_dir: Path | None = None) -> list[dict]:
    """Backlog items this eval has filed: id and title, oldest first.

    Keyed on the `youtube-eval` tag in the front matter. Read from the head
    of each file only — the board is ~450 files and this runs once per turn.
    """
    d = backlog_dir or BACKLOG_DIR
    out: list[dict] = []
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.md")):
        m = re.match(r"^(\d+)[-_]", f.name)
        if not m:
            continue
        head = _read_head(f)
        if not head.startswith("---"):
            continue
        fm_end = head.find("\n---", 3)
        fm = head[:fm_end] if fm_end != -1 else head
        if not re.search(rf"^-\s*{EVAL_TAG}\s*$", fm, re.M) and f"'{EVAL_TAG}'" not in fm and f"{EVAL_TAG}," not in fm:
            continue
        body = head[fm_end:] if fm_end != -1 else ""
        t = re.search(r"^#\s+(.+)$", body, re.M)
        out.append({"id": int(m.group(1)), "title": (t.group(1).strip() if t else f.stem)[:120]})
    out.sort(key=lambda t: t["id"])
    return out


def _filed_item_exists(item_id: int, video_id: str, backlog_dir: Path | None = None) -> bool:
    """A `FILED: #n` claim holds only if that file exists, carries the eval
    tag, and mentions this video — the shape the prompt asked for."""
    d = backlog_dir or BACKLOG_DIR
    for f in d.glob(f"{int(item_id)}-*.md"):
        text = _read_head(f, 20000)
        return EVAL_TAG in text and video_id in text
    return False


def _vault_dirty_paths() -> set[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(VAULT_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line[3:].strip().strip('"') for line in out.splitlines() if line[3:].strip()}


def _unexpected_vault_writes(before: set[str]) -> list[str]:
    """Paths this turn changed in the vault outside the prefixes it should.

    A diff against a baseline, never a snapshot: the vault is never clean
    (the scheduler rewrites a task file on every run), so a snapshot reports
    everyone else's churn. Reported, not reverted.
    """
    changed = _vault_dirty_paths() - set(before)
    return sorted(p for p in changed if not p.startswith(EXPECTED_VAULT_PREFIXES))


# ── The script ───────────────────────────────────────────────────────────────


async def _script(channel: str, *args: str, timeout: float) -> dict:
    """Run the monitor script for one channel and return its JSON line.

    A subprocess, not an import: the script shells out to uv/yt-dlp with
    blocking calls, and the pool runs on the backend's one event loop. It is
    stdlib-only, so the backend's interpreter runs it fine.
    """
    cmd = [sys.executable, str(SCRIPT), "--channel", channel, *args, "--json"]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ScriptError(f"{' '.join(args[:2])} exceeded {timeout:.0f}s") from None
    text = out.decode("utf-8", "replace")
    for line in reversed(text.splitlines()):
        if line.startswith(JSON_MARK):
            try:
                return json.loads(line[len(JSON_MARK):])
            except json.JSONDecodeError as exc:
                raise ScriptError(f"{args[0]}: bad JSON line: {exc}") from None
    tail = (err.decode("utf-8", "replace").strip() or text.strip())[-600:]
    raise ScriptError(f"{args[0]} rc={proc.returncode}: {tail}")


# ── Scheduling ───────────────────────────────────────────────────────────────


def _queued_for_source(queue: WorkQueue) -> int:
    try:
        depth = queue.depth_by_source().get(NAME, {}) or {}
    except Exception:  # noqa: BLE001 — a depth read must never stop enqueueing
        return 0
    return sum(int(v) for k, v in depth.items() if k in ("queued", "claimed"))


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    """Keep a small, interleaved slice of each channel's pending videos queued.

    `batch` bounds what sits in the queue at once, not what gets done: the
    tick refills to that level. Channels are interleaved so a channel with
    280 videos in its window does not push a channel with 55 to the end of
    the day. `--register-new` also picks up new uploads on every tick.
    """
    channels = [str(c) for c in (src_cfg.get("channels") or DEFAULT_CHANNELS)]
    batch = int(src_cfg.get("batch", 3))
    priority = int(src_cfg.get("priority", DEFAULT_PRIORITY))
    register_timeout = float(src_cfg.get("register_timeout_seconds", 300))
    max_turns = int(src_cfg.get("max_turns", 40))

    room = batch - _queued_for_source(queue)
    if room <= 0:
        return

    per_channel: dict[str, list[dict]] = {}
    for channel in channels:
        try:
            out = await _script(channel, "--register-new", timeout=register_timeout)
        except ScriptError as exc:
            logger.warning("youtube-digest: %s --register-new failed: %s", channel, exc)
            continue
        per_channel[channel] = list(out.get("pending") or [])

    cursors = {c: 0 for c in per_channel}
    while room > 0 and any(cursors[c] < len(per_channel[c]) for c in per_channel):
        for channel in channels:
            rows = per_channel.get(channel) or []
            if cursors.get(channel, 0) >= len(rows) or room <= 0:
                continue
            row = rows[cursors[channel]]
            cursors[channel] += 1
            new_id = queue.enqueue(
                source=NAME,
                kind="video",
                payload={"channel": channel, "video_id": row["video_id"],
                         "title": row.get("title", ""), "published": row.get("published", ""),
                         "max_turns": max_turns},
                priority=priority,
                dedup_key=f"{NAME}:{channel}:{row['video_id']}",
            )
            if new_id is not None:
                room -= 1
                logger.info("Enqueued youtube-digest %s %s: %s",
                            channel, row["video_id"], (row.get("title") or "")[:60])


# ── Execution ────────────────────────────────────────────────────────────────


def _eval_record(parsed: Optional[dict], session_id: str) -> dict:
    parsed = parsed or {}
    return {
        "relevance": parsed.get("relevance"),
        "verdict": parsed.get("verdict"),
        "areas": parsed.get("areas") or [],
        "source_kind": parsed.get("source_kind"),
        "approach": parsed.get("approach"),
        "idea": parsed.get("idea") or "",
        "duplicate_of": parsed.get("duplicate_of"),
        "filed": parsed.get("filed"),
        "note_result": parsed.get("result"),
        "session_id": session_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _fail(channel: str, video_id: str, why: str) -> None:
    try:
        await _script(channel, "--fail", video_id, "--reason", why[:400], timeout=60)
    except ScriptError as exc:
        logger.warning("youtube-digest: could not record failure for %s %s: %s", channel, video_id, exc)


async def execute(item: QueueItem) -> dict[str, Any]:
    from workers.sources import get_sources_config

    payload = item.payload or {}
    channel = str(payload.get("channel") or "")
    video_id = str(payload.get("video_id") or "")
    if not channel or not video_id:
        return {"status": "failed", "summary": "queue item carries no channel/video_id"}
    src_cfg = get_sources_config().get(NAME, {}) or {}
    base_meta = {"channel": channel, "video_id": video_id}

    # 1. The bundle. A transcript that cannot be fetched is recorded by the
    #    script itself (`--fetch` marks the attempt), so nothing to do here
    #    but report it.
    try:
        fetched = await _script(channel, "--fetch", video_id,
                                timeout=float(src_cfg.get("fetch_timeout_seconds", 420)))
    except ScriptError as exc:
        return {"status": "failed", "summary": f"{channel} {video_id}: fetch crashed: {exc}"[:500],
                "meta": {**base_meta, "fetch_crashed": True}}
    if not fetched.get("ok"):
        why = str(fetched.get("error") or "fetch failed")
        return {"status": "failed", "summary": f"{channel} {video_id}: {why}"[:500],
                "meta": {**base_meta, "fetch_error": why,
                         "failure_count": fetched.get("failure_count")}}
    meta = fetched["meta"]
    note_path = Path(meta["target_note"])
    title = str(meta.get("title") or video_id)

    # 2. The session.
    tracked = await asyncio.to_thread(tracked_items)
    prompt = build_prompt(meta, tracked)
    vault_before = await asyncio.to_thread(_vault_dirty_paths)
    try:
        run = await run_prompt_in_session(
            prompt, title=f"{meta.get('channel_name', channel)}: {title[:52]}",
            source=NAME,
            max_turns=int(payload.get("max_turns") or src_cfg.get("max_turns", 40)),
            priority=1,
            inner_voice=bool(src_cfg.get("inner_voice", True)),
            extra_disallowed=list(DISALLOWED))
    except DrainActive as exc:
        # Not the video's fault: the entry stays `fetched` and is re-offered.
        return {"status": "skipped", "summary": f"landing in progress: {exc}"[:500],
                "meta": {**base_meta, "drain_active": True}}
    except TurnTimeout as exc:
        await _fail(channel, video_id, f"turn timeout: {exc}")
        return {"status": "failed", "summary": f"{channel} {video_id}: {exc}"[:500],
                "meta": {**base_meta, "turn_timeout": True}}

    session_id = run.get("session_id")
    text = run.get("text") or ""
    parsed = parse_result(text)
    on_disk = await asyncio.to_thread(_note_is_real, note_path, video_id)
    unexpected = await asyncio.to_thread(_unexpected_vault_writes, vault_before)

    # 3. Disk decides.
    if not on_disk:
        why = (f"turn ended ({run.get('stop_reason')}, {run.get('num_turns')} iterations) "
               f"without a note at {note_path.name}")
        await _fail(channel, video_id, why)
        return {"status": "failed", "summary": f"{channel} {video_id}: {why}"[:500],
                "response": text,
                "meta": {**base_meta, "session_id": session_id,
                         "stop_reason": run.get("stop_reason"),
                         "empty_response": not text.strip(),
                         "unexpected_vault_writes": unexpected}}

    eval_result = _eval_record(parsed, session_id)
    if eval_result.get("filed"):
        ok = await asyncio.to_thread(_filed_item_exists, int(eval_result["filed"]), video_id)
        if not ok:
            eval_result["filed_unverified"] = eval_result.pop("filed")
            logger.warning("youtube-digest: %s %s claimed FILED #%s but no such item is on disk",
                           channel, video_id, eval_result["filed_unverified"])

    # 4. Record, then refresh the report note (best effort).
    try:
        await _script(channel, "--complete", video_id, "--note", str(note_path),
                      "--eval-json", json.dumps(eval_result), timeout=60)
    except ScriptError as exc:
        # The note exists and the verdict is in this run record; a second
        # attempt would redo the whole session, so this is a warning.
        logger.warning("youtube-digest: --complete failed for %s %s: %s", channel, video_id, exc)
    try:
        await _script(channel, "--eval-report", timeout=60)
    except ScriptError as exc:
        logger.warning("youtube-digest: --eval-report failed for %s: %s", channel, exc)

    verdict = eval_result.get("verdict") or "no verdict"
    rel = eval_result.get("relevance")
    bits = [f"{meta.get('channel_name', channel)}: {title[:60]} — {verdict}"
            + (f" ({rel})" if rel is not None else "")]
    if eval_result.get("filed"):
        bits.append(f"filed #{eval_result['filed']}")
    elif eval_result.get("filed_unverified"):
        bits.append(f"claimed #{eval_result['filed_unverified']} (unverified)")
    elif eval_result.get("duplicate_of"):
        bits.append(f"duplicate of #{eval_result['duplicate_of']}")
    if parsed and parsed.get("result") == "kept":
        bits.append("note kept")
    if unexpected:
        bits.append(f"{len(unexpected)} unexpected vault write(s)")
    return {
        "status": "success",
        "summary": ", ".join(bits)[:500],
        "artifact_path": str(note_path),
        "response": text,
        "meta": {**base_meta, "session_id": session_id, "note": str(note_path),
                 "stop_reason": run.get("stop_reason"), "num_turns": run.get("num_turns"),
                 "eval": eval_result, "unexpected_vault_writes": unexpected,
                 "parsed": parsed is not None},
    }
