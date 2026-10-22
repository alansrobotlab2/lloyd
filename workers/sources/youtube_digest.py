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

Four properties worth knowing before changing anything here:

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
* **The protocol is not in this file.** `SKILL` names the vault skill —
  `~/obsidian/skills/youtube-digest/SKILL.md` — which `build_prompt` reads at
  prompt-build time and renders with the per-bundle `TASK_BLOCK` (#737). So a
  change to the steps, the prose ban-list or the adoption rules is a vault
  edit that takes effect on the next run, not a code round. What is here is
  the placeholders and the note header, and a missing skill is a named failure
  rather than a fallback prompt.

* **A paper a video cites is a second kind of row** (`kind="paper"`), read
  in its own session. The script registers a paper when the bundle names an
  arXiv id, and again after the video's turn for the papers its RESULT block
  names (`PAPERS:` — most Discover AI descriptions give a title and authors,
  no id, so the session is the only reader that can name them). `execute`
  dispatches on the kind; `_execute_paper` is the same shape as the video
  path, with its own vault skill (`PAPER_SKILL`) and no backlog filing.

Open-source frameworks and tools may be proposed for direct adoption or
evaluation. Commercial products may not — the eval names the aspects worth
recreating locally instead. That rule is Alan's, and the session is told it in
step 3 of the skill.
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

from app.paths import LLOYD_HOME, VAULT_PENDING_RESEARCH_DIR, VAULT_ROOT
from workers.queue import WorkQueue, QueueItem
from workers.sources._common import (
    WORKER_AUTOMOD_BAN, DrainActive, TurnTimeout, _worker_run_options,
    build_skill_prompt, reconcile_fact_writes, run_prompt_in_session,
)

logger = logging.getLogger("lloyd-workers.youtube-digest")

NAME = "youtube-digest"
#: Between backlog triage (55) and the research stream (70). A digest is
#: routine; it should not starve an automod round, and it should run before
#: the next research topic because Alan reads these.
DEFAULT_PRIORITY = 60

#: How long an infra-shaped failure ("turn produced nothing") parks ITS OWN
#: queue row before the video is digested again — the bound #1714 exists to
#: state. The no-penalty half of that branch is deliberate and stays: it never
#: calls the script's `--fail`, so `seen.json` records nothing against the video
#: and the bundle survives (on 2026-09-09 a primary restart put six videos
#: through it in five seconds each, and all six went on to work). What was NOT
#: deliberate is that nothing spaced the re-offer: the row stayed `fetched`, the
#: source's tick is `interval_seconds` 300, and every tick enqueued a NEW row
#: with `attempts` back to 1. Measured on 2026-09-24, ids 409/410/411 enqueued at
#: 13:56:39 and the same three videos again as 413/414/415 at 14:01:49 — 300.00 s,
#: with `not_before` NULL on all 53 youtube-digest rows ever written.
#:
#: 900 s is the scanner path's own `RETRY_INTERVAL_SECONDS`
#: (`scripts/youtube_channel_monitor.py:139`, and the same 900 the `failed`
#: registry path uses via `_is_retry_eligible`), so a video that lost a turn to
#: the engine is offered again on the clock the system already treats as one
#: retry's worth of spacing — at most twice more, since the row now survives its
#: run and the queue's `max_attempts` (3) stops handing it out at the third
#: claim. A brief restart therefore costs a video 15 minutes of lateness and
#: exactly nothing in `seen.json`.
INFRA_DEFER_SECONDS = 900

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
    *WORKER_AUTOMOD_BAN,
    "Bash", "Edit", "Task",
    "http_request",
    "browser_evaluate", "browser_fill", "browser_click",
    "browser_press", "browser_cookies", "browser_drag", "browser_select",
    "autonomy_write_task", "autonomy_delete_task", "autonomy_run_task",
    "autonomy_config", "autonomy_config_set",
    "research_propose", "research_next", "research_complete",
)

VERDICTS = ("actionable", "worth_a_look", "background", "not_relevant")
SOURCE_KINDS = ("open-source", "commercial", "paper", "concept")
APPROACHES = ("adopt", "recreate", "experiment", "read", "none")
AREAS = ("model", "inference", "harness", "tools", "memory", "knowledge-graph",
         "retrieval", "skills", "autonomy", "automod", "eval", "voice", "research", "ui")
_RESULTS = ("written", "kept", "failed")
_FIELD_RE = re.compile(
    r"^(RESULT|NOTE|RELEVANCE|VERDICT|AREAS|SOURCE_KIND|APPROACH|IDEA|DUPLICATE_OF|FILED|PAPERS):\s*(.*)$",
    re.I)
_ID_RE = re.compile(r"#?\s*(\d+)")

#: The RESULT block as the harness finalizer's schema (app/harness/finalizer.py).
#: Every enum is the tuple the text parser validates against, so the two paths
#: cannot disagree on a vocabulary; an outcome outside `_RESULTS` is a schema
#: rejection, not a field that quietly comes back None. The id fields stay
#: strings ("#123" or "none") and go through the same `_id_or_none` as the text.
#: Its string leaves are capped — see the block below the three constants, which
#: says what sizes each one. What this line claimed before #2444 was that no
#: `maxLength` may appear, the same absolute ban #2240 retired for `REVIEW_SCHEMA`
#: and for the same reason: an open string is what makes the finalizer blame
#: `harness.finalizer.max_tokens` for a cut it cannot be the cause of.
DIGEST_NOTE_MAX = 400
DIGEST_IDEA_MAX = 400
DIGEST_ID_MAX = 20
#: `PAPERS:` is at most MAX_PAPERS references on one line, each an arXiv id or
#: a title. 600 holds four 150-character titles; `_paper_refs` drops a fifth.
DIGEST_PAPERS_MAX = 600
MAX_PAPERS = 4
PAPER_KIND = "paper"
#: A stub is ~1.5 kB of abstract; a note that read the paper is several times that.
_MIN_PAPER_NOTE_BYTES = 2500

# Bounded grammar (#2444): `note`/`idea` at DIGEST_NOTE_MAX / DIGEST_IDEA_MAX (400,
# one line each) and the two id fields at DIGEST_ID_MAX (20). `_shape` slices none of
# them, so no cap sits above a Python slice; it only ends a runaway field inside
# `harness.finalizer.max_tokens` (8192), which moves a degenerate draw onto the bounded
# branch — "generation diverged at N tokens … not a budget" (`DIVERGENCE_MARKER`) plus
# its one re-draw — instead of the unbounded branch's advice to raise that knob
# (#1706). What sizes them is `workers.db`, and the JSON path is what the sentence this
# replaces got wrong: a run's structured-ness is `meta_json`'s TOP-level
# `verdict_source`, not a stored finalizer object, so
# `select count(*) from runs where source='youtube-digest'
#    and json_extract(meta_json,'$.verdict_source')='structured'
#    and (json_extract(meta_json,'$.structured_error') is null
#         or json_extract(meta_json,'$.structured_error')='')` → 221 of 372 runs,
# measured 2026-10-22. The schema path is therefore the live path, and
# `select max(length(json_extract(meta_json,'$.eval.idea'))), count(*) from runs
#    where source='youtube-digest' and json_extract(meta_json,'$.eval.idea') is not null`
# → 314 over 236 runs against DIGEST_IDEA_MAX (400), 0 of the 236 at or over it. The
# other half is not in `meta_json` at all — `$.eval` has no `note` key — so `note` is
# sized off the `NOTE:` line of `runs.response_json`, counted by
# `select count(*) from runs where source='youtube-digest' and response_json like
#    '%NOTE:%'`
# → 318 of the 372 runs (317 earlier the same day, which is the denominator #3147 was
# filed with; the pattern is `%NOTE:%` and not `NOTE:%` because the label is not at the
# start of that column, and the prefix form returns 0). SQL slices no lines, so the
# maxima come off the `note_line` field of the committed projection named below: the
# line's content after its label reaches 152 characters and the value `_shape`'s
# `one("NOTE")` keeps (that content, whitespace-collapsed and stripped of `()`) reaches
# 151, both
# against DIGEST_NOTE_MAX (400) with 0 at or over; the raw line including the label is
# two characters longer, 157, which is why #2444 and #3147 quote different numbers for
# one corpus. `$.eval.verdict_source`, the path that would have been guessed here,
# is NULL on all 372 — which is how "no `youtube-digest` run stores a structured verdict"
# stood in this block for a month although it was false the day it was written: 108 of
# the 142 runs completed by 2026-10-09 already carried `structured`, the first landing
# 2026-09-24. DIGEST_ID_MAX is inert on its own two leaves rather than unmeasured:
# `$.eval.duplicate_of` and `$.eval.filed` return integers or NULL and never a string (1
# run both, 80 the first alone, 47 the second alone, 244 neither), so a `maxLength` there
# is grammar for a value this store never writes. Every figure above is a corpus that
# grows nightly — the note denominator moved 317 to 318 inside 2026-10-22 itself — so
# what is pinned is the projection behind them,
# `tests/fixtures/workers_youtube_digest_capped_fields_2026-10-22-item3147.jsonl`: one row
# per `youtube-digest` run (372 on 2026-10-22) carrying `verdict_source`,
# `structured_error`, `idea`, `duplicate_of`, `filed` and that run's `NOTE:` line, from
# which `tests/test_automod_schema_bounds.py` re-derives each number in this block — the
# live rows are 1.6 MB of `response_json` and 0.3 MB of `meta_json` and a test may not
# read the running store. What is still true of the ledger is the narrow claim only: it
# has no
# digest event kind, 0 rows named `digest` or `youtube` over it and its two folded
# archives on 2026-10-22 (#2444/2, #3147).
RESULT_SCHEMA: dict = {
    "type": "object",
    "title": "youtube_digest_result",
    "properties": {
        "result": {"type": "string", "enum": list(_RESULTS)},
        "note": {"type": "string", "maxLength": DIGEST_NOTE_MAX},
        # Clamped to 0-100 in `_shape`, like the text path, not in the grammar.
        "relevance": {"type": "integer"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "areas": {"type": "array", "items": {"type": "string", "enum": list(AREAS)}},
        "source_kind": {"type": "string", "enum": list(SOURCE_KINDS)},
        "approach": {"type": "string", "enum": list(APPROACHES)},
        "idea": {"type": "string", "maxLength": DIGEST_IDEA_MAX,
                "description": "One line naming the specific thing, or none."},
        "duplicate_of": {"type": "string", "maxLength": DIGEST_ID_MAX,
                "description": "#id or none"},
        "filed": {"type": "string", "maxLength": DIGEST_ID_MAX,
                "description": "#id or none"},
        "papers": {"type": "string", "maxLength": DIGEST_PAPERS_MAX,
                "description": "arXiv ids or exact titles separated by ' | ', or none"},
    },
    "required": ["result", "note", "relevance", "verdict", "areas", "source_kind",
                 "approach", "idea", "duplicate_of", "filed", "papers"],
    "additionalProperties": False,
}


class ScriptError(RuntimeError):
    """The monitor script did not produce its JSON line."""


# ── The prompt ───────────────────────────────────────────────────────────────
#
# The protocol — the numbered steps, the shared prose ban-list, the
# write-the-note-then-read-it-back rule and the tail check — is the vault skill
# named by `SKILL` below, loaded at prompt-build time (#737). What stays here is
# the per-bundle data only: one `.format()` template with 22 fields, and the note
# header, whose values nothing but this source knows. Splitting on the
# placeholder boundary is what makes a protocol fix a vault edit rather than a
# code round, and it leaves exactly one brace surface: this template. The skill
# body is handed over unformatted, so a brace written there would reach the model
# as a literal — `tests/test_youtube_digest_source.py` pins the field set, the
# absence of unrendered fields, and that the skill stays brace-free.

SKILL = "youtube-digest"

TASK_BLOCK = """\
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
measurements: {measurements_path} (live numbers captured at fetch time: {measurements_summary})
</video>

## Your task

One video, one note. Write it to exactly the path below, then evaluate it for
Lloyd and answer with the RESULT block at the end:
   {target_note}
{existing_note_instruction}

Start the file with exactly this header, values filled in, and these sections in
this order — the source recognises the note on disk by the front matter, and the
tail check depends on `## Open Questions` coming last:

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

Already tracked by this eval — compare against these before filing, and if the
idea is the same do not file, set DUPLICATE_OF instead:
{tracked}

Filing tags for step 4: `{eval_tag}`, `{channel_key}`, plus the area tags.
Source line for the item's description:
   **Source:** {channel_name} — "{title}" — {url} (published {published}); vault note [[{note_stem}]]

Profile to read for step 3: {profile_path}

`PAPERS` in the block below names the research papers this video is about, at \
most four: the ones it presents or walks through, not every work it mentions \
in passing. Give the arXiv id where the description or transcript has one, \
otherwise the paper's full title exactly as the description prints it. Each \
one is read in full by a later run and gets its own note, so a title you are \
unsure of is left out.

End your final message with exactly this block and nothing after it:

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
PAPERS: <arXiv ids or exact paper titles, separated by " | ", or none>

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


class SkillProtocolMissing(RuntimeError):
    """The vault skill carrying the digest protocol is absent or empty.

    Named so the run record says "the skill is gone" instead of describing a
    digest turn that answered with nothing to follow. There is deliberately no
    fallback prompt: the second copy of the rules that would make a fallback
    possible is the defect #737 was filed for, and it fails silently — the run
    still writes a note, just to rules no one is editing any more.
    """


def load_skill(slug: str = SKILL) -> str:
    """Read the digest protocol out of the vault skill.

    The same route `workers/sources/deep_research.py` takes:
    `autonomy._load_skill_content` resolves the slug to
    `~/obsidian/skills/<slug>/SKILL.md`. Read per call and never cached — the
    whole point of the move is that a vault edit changes the next run.
    """
    from app import autonomy

    text = autonomy._load_skill_content(slug) or ""
    if not text.strip():
        raise SkillProtocolMissing(
            f"skills/{slug}/SKILL.md is missing or empty: the digest protocol is "
            "not in this file any more, so there is nothing to send the session")
    return text


def build_prompt(meta: dict, tracked: list[dict], *, skill_text: Optional[str] = None) -> str:
    """Render the digest prompt for one bundle: the vault skill plus its task.

    `tracked` is the list of backlog items this eval has already filed, so a
    channel that returns to a theme every week (Discover AI and "the harness",
    say) does not file it every week.

    `skill_text` is a seam for tests; production leaves it None and reads the
    vault, which is the fact this split exists to establish.
    """
    enrichment = meta.get("enrichment") or {}
    notes = []
    for r in list(enrichment.get("github") or []) + list(enrichment.get("papers") or []):
        p = (r or {}).get("note_path")
        if p:
            notes.append(Path(p).stem)
    tracked_lines = [f"   - #{t['id']} {t['title']}" for t in tracked] or ["   - none yet"]
    target = str(meta.get("target_note") or "")
    task_block = TASK_BLOCK.format(
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
        measurements_path=meta.get("measurements_path") or "none",
        measurements_summary=meta.get("measurements_summary") or "none captured",
        profile_path=str(PROFILE_PATH),
        tracked="\n".join(tracked_lines),
        eval_tag=EVAL_TAG,
        areas=", ".join(AREAS),
    )
    return build_skill_prompt(load_skill() if skill_text is None else skill_text,
                              job=NAME, task_block=task_block)


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
    the note is the primary product and is verified on disk separately. On the
    finalizer path (`parse_verdict`) an outcome outside `_RESULTS` is a schema
    rejection instead, and `execute` no longer counts it as a turn that ran.
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

    return _shape(lambda key: " ".join(fields.get(key, [])))


def _shape(raw) -> dict:
    """One set of clamps for both paths: `raw(FIELD)` is the field's text."""
    def one(key: str) -> str:
        return " ".join(str(raw(key) or "").split()).strip().strip("`'\"")

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
    papers = [" ".join(p.split()).strip("`'\" ") for p in str(raw("PAPERS") or "").split("|")]
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
        "papers": [p for p in papers if p and p.lower() != "none"][:MAX_PAPERS],
    }


def parse_verdict(text: str, structured: Optional[dict] = None) -> Optional[dict]:
    """The turn's outcome, from the finalizer's object or from the RESULT block.

    `autotriage.parse_verdict`'s rule: the object wins when it carries a known
    result, and the block stays in the prompt regardless, because the
    finalizer is skipped on any turn that did not end of its own accord and
    can fail on one that did. `source` names the path that answered.
    """
    if isinstance(structured, dict):
        def raw(key: str):
            value = structured.get(key.lower())
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
            return "" if value is None else str(value)
        parsed = _shape(raw)
        if parsed["result"] is not None:
            return {**parsed, "source": "structured"}
    parsed = parse_result(text)
    return {**parsed, "source": "regex"} if parsed is not None else None


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
    tag, and mentions this video — the shape the prompt asked for.

    Or, since `youtube-eval` writes merge like the loop's own (2026-09-14),
    if the tool folded the filing into an existing item: that item need not
    carry the eval tag, and the merged text sits at the END of its file under
    a "Merged finding" heading, past any head read. So the whole file is read
    for that case."""
    d = backlog_dir or BACKLOG_DIR
    for f in d.glob(f"{int(item_id)}-*.md"):
        text = _read_head(f, 20000)
        if EVAL_TAG in text and video_id in text:
            return True
        whole = _read_head(f, 2_000_000)
        return "## Merged finding" in whole and video_id in whole.split("## Merged finding", 1)[1]
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

    **Every option value must ride attached to its option** (`--fetch=<id>`),
    never as a separate token. A YouTube id is base64url, so roughly one in
    forty starts with `-`, and argparse reads a bare `-tviRdpmHvs` as a flag:
    the script exits 2 with its usage text before it has done anything. That
    is not a video that failed, it is a video that can never be attempted —
    and because `--fail` took its id the same way, the failure could not be
    recorded either, so the row stayed `pending` and was re-offered every
    tick. On 2026-09-09 seven such ids sat at the head of the two channels'
    newest-first queues and burned 89 runs; Discover AI had one at its head
    and stopped draining. `tests/test_youtube_digest_source.py` pins the
    shape of every argv this builds.
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

    # Papers first, `papers.batch` of them: the channels always have a video
    # pending, so a paper offered after them would never get the room. The
    # registry hands them out oldest first, and the dedup key makes a paper
    # that is already queued or running cost nothing here.
    papers = _papers_cfg(src_cfg)
    if papers["enabled"]:
        try:
            out = await _script(channels[0], "--papers-pending", timeout=60)
        except ScriptError as exc:
            logger.warning("youtube-digest: --papers-pending failed: %s", exc)
            out = {}
        offered = [r for r in out.get("pending") or [] if r.get("arxiv_id")]
        for row in offered[:max(0, papers["batch"])]:
            if room <= 0:
                break
            new_id = queue.enqueue(
                source=NAME, kind=PAPER_KIND,
                payload={"arxiv_id": row["arxiv_id"], "title": row.get("title", ""),
                         "max_turns": papers["max_turns"]},
                priority=priority,
                dedup_key=f"{NAME}:{PAPER_KIND}:{row['arxiv_id']}",
            )
            if new_id is not None:
                room -= 1
                logger.info("Enqueued youtube-digest paper %s: %s",
                            row["arxiv_id"], (row.get("title") or "")[:60])

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
                         "max_turns": max_turns,
                         # Kill switch, in the payload like autotriage's: off,
                         # the turn runs identically and only the block is read.
                         "structured_verdict": bool(src_cfg.get("structured_verdict", True))},
                priority=priority,
                dedup_key=f"{NAME}:{channel}:{row['video_id']}",
            )
            if new_id is not None:
                room -= 1
                logger.info("Enqueued youtube-digest %s %s: %s",
                            channel, row["video_id"], (row.get("title") or "")[:60])


# ── Papers ───────────────────────────────────────────────────────────────────
#
# One session per cited paper, after the video that cited it. The script has
# already put the full text on disk and an abstract-only note at the target
# path; the session replaces that note with one written from the paper.
#
# Same split as the video prompt: the protocol is the vault skill named by
# `PAPER_SKILL` — `~/obsidian/skills/paper-digest/SKILL.md`, read at
# prompt-build time and never formatted — and what is here is the per-paper
# task block. A missing skill fails the run by name and is not counted against
# the paper.

PAPER_SKILL = "paper-digest"

PAPER_TASK_BLOCK = """\
<paper>
arxiv_id: {arxiv_id}
title: {title}
authors: {authors}
submitted: {published}
url: {url}
text: {text_path} ({text_words} words, {text_lines} lines, from arXiv's {text_source}){truncated}
metadata: {meta_path}
</paper>

## Your task

One paper, one note. Replace the placeholder at exactly this path:
   {target_note}

Start the file with exactly this header (add your topic tags under `tags:`):

{note_header}
End the file with exactly this section:

## Discussed in

{discussed}

Profile to read for `## Relevance to Lloyd`: {profile_path}

End your final message with exactly this block and nothing after it:

RESULT: <written|failed>
NOTE: {target_note}

`written` means the full note is at the path above; `failed` means you could \\
not produce it — say why in one line before the block.
"""

_PAPER_TRUNCATED = "\n   The text was cut at its size cap; say in `## Limitations` that the tail was not read."

#: The paper turn writes one note and files nothing: the video's turn already
#: judged the idea, and a second filing per paper is the loop opening more
#: than it closes.
PAPER_DISALLOWED: tuple[str, ...] = (*DISALLOWED, "backlog_write_task", "fact_add")


def build_paper_prompt(meta: dict, *, skill_text: Optional[str] = None) -> str:
    task_block = PAPER_TASK_BLOCK.format(
        arxiv_id=meta.get("arxiv_id", ""),
        title=meta.get("title", ""),
        authors=", ".join(meta.get("authors") or []) or "unknown",
        published=meta.get("published", "") or "unknown",
        url=meta.get("url", ""),
        text_path=meta.get("text_path", ""),
        text_words=meta.get("text_words", "?"),
        text_lines=meta.get("text_lines", "?"),
        text_source="PDF" if meta.get("text_source") == "pdf" else "HTML rendering",
        truncated=_PAPER_TRUNCATED if meta.get("text_truncated") else "",
        meta_path=meta.get("meta_path", ""),
        target_note=str(meta.get("target_note") or ""),
        note_header=meta.get("note_header", ""),
        discussed="\n".join(meta.get("discussed") or ["- None recorded"]),
        profile_path=str(PROFILE_PATH),
    )
    return build_skill_prompt(load_skill(PAPER_SKILL) if skill_text is None else skill_text,
                              job=NAME, task_block=task_block)


def _paper_note_is_real(path: Path, arxiv_id: str) -> bool:
    """The full note for this paper, and not the placeholder it replaces.

    The placeholder carries `digest: abstract`, so `digest: full` on a file
    the script wrote minutes ago is proof this turn wrote it — the stronger
    form of the video path's "a pre-existing note proves nothing". The tail is
    checked because a write that died mid-file leaves a good header.
    """
    try:
        if not path.is_file():
            return False
        text = path.read_text(errors="replace")
    except OSError:
        return False
    if len(text.encode()) < _MIN_PAPER_NOTE_BYTES or not text.startswith("---"):
        return False
    end = text.find("\n---", 3)
    head = text[:end] if end != -1 else text[:4000]
    return bool(re.search(rf"^arxiv_id:\s*['\"]?{re.escape(arxiv_id)}['\"]?\s*$", head, re.M)
                and re.search(r"^digest:\s*full\s*$", head, re.M)
                and re.search(r"^## Discussed in\s*$", text, re.M))


def _papers_cfg(src_cfg: dict) -> dict:
    cfg = src_cfg.get("papers")
    cfg = cfg if isinstance(cfg, dict) else {}
    return {"enabled": bool(cfg.get("enabled", True)), "batch": int(cfg.get("batch", 1)),
            "max_turns": int(cfg.get("max_turns", src_cfg.get("max_turns", 40)))}


async def _paper_fail(arxiv_id: str, why: str) -> None:
    try:
        await _script(DEFAULT_CHANNELS[0], f"--paper-fail={arxiv_id}", f"--reason={why[:400]}", timeout=60)
    except ScriptError as exc:
        logger.warning("youtube-digest: could not record failure for paper %s: %s", arxiv_id, exc)


async def _execute_paper(item: QueueItem) -> dict[str, Any]:
    from workers.sources import get_sources_config

    payload = item.payload or {}
    arxiv_id = str(payload.get("arxiv_id") or "")
    if not arxiv_id:
        return {"status": "failed", "summary": "paper queue item carries no arxiv_id"}
    src_cfg = get_sources_config().get(NAME, {}) or {}
    base_meta = {"kind": PAPER_KIND, "arxiv_id": arxiv_id}
    label = f"paper {arxiv_id}"

    # 1. The bundle. As for a video: a text that cannot be fetched is counted by
    #    the script, a crash is counted here.
    try:
        fetched = await _script(DEFAULT_CHANNELS[0], f"--paper-fetch={arxiv_id}",
                                timeout=float(src_cfg.get("fetch_timeout_seconds", 420)))
    except ScriptError as exc:
        await _paper_fail(arxiv_id, f"fetch crashed: {exc}")
        return {"status": "failed", "summary": f"{label}: fetch crashed: {_one_line(exc)}",
                "meta": {**base_meta, "fetch_crashed": True}}
    if not fetched.get("ok"):
        why = str(fetched.get("error") or "fetch failed")
        return {"status": "failed", "summary": f"{label}: {_one_line(why)}",
                "meta": {**base_meta, "fetch_error": why,
                         "failure_count": fetched.get("failure_count")}}
    meta = fetched["meta"]
    note_path = Path(meta["target_note"])
    title = str(meta.get("title") or arxiv_id)

    # 2. The session.
    try:
        prompt = await asyncio.to_thread(build_paper_prompt, meta)
    except SkillProtocolMissing as exc:
        # The deploy's fault, not the paper's: the row stays `fetched`.
        logger.error("youtube-digest: %s", exc)
        return {"status": "failed", "summary": f"{label}: {_one_line(exc)}",
                "meta": {**base_meta, "skill_missing": True}}
    vault_before = await asyncio.to_thread(_vault_dirty_paths)
    try:
        run = await run_prompt_in_session(
            prompt, title=f"Paper: {title[:58]}", source=NAME,
            max_turns=int(payload.get("max_turns") or _papers_cfg(src_cfg)["max_turns"]),
            priority=1, extra_disallowed=list(TURN_ENVELOPES["paper-note-turn"]))
    except DrainActive as exc:
        return {"status": "skipped", "summary": f"landing in progress: {_one_line(exc)}",
                "meta": {**base_meta, "drain_active": True}}
    except TurnTimeout as exc:
        await _paper_fail(arxiv_id, f"turn timeout: {exc}")
        return {"status": "failed", "summary": f"{label}: {_one_line(exc)}",
                "meta": {**base_meta, "turn_timeout": True}}

    session_id = run.get("session_id")
    text = run.get("text") or ""
    stop_reason = run.get("stop_reason")
    on_disk = await asyncio.to_thread(_paper_note_is_real, note_path, arxiv_id)
    unexpected = await asyncio.to_thread(_unexpected_vault_writes, vault_before)
    run_meta = {**base_meta, "session_id": session_id, "stop_reason": stop_reason,
                "num_turns": run.get("num_turns"), "unexpected_vault_writes": unexpected}

    # 3a. Infra-shaped, as in the video path: not the paper's fault, parked on
    #     its own row for INFRA_DEFER_SECONDS and never counted.
    if not text.strip() and stop_reason is None and not on_disk:
        errs = [str(e)[:160] for e in (run.get("errors") or [])[:2]]
        why = f"turn produced nothing ({'; '.join(errs) or 'no error reported'})"
        logger.warning("youtube-digest: %s: %s — left fetched, retried in %ds",
                       label, why, INFRA_DEFER_SECONDS)
        return {"status": "failed", "summary": f"{label}: {_one_line(why)}",
                "defer_seconds": INFRA_DEFER_SECONDS,
                "meta": {**run_meta, "infra": True, "empty_response": True}}

    # 3b. Disk decides.
    if not on_disk:
        why = (f"turn ended ({stop_reason}, {run.get('num_turns')} iterations) "
               f"without a full note at {note_path.name}")
        await _paper_fail(arxiv_id, why)
        return {"status": "failed", "summary": f"{label}: {_one_line(why)}",
                "response": text, "meta": {**run_meta, "empty_response": not text.strip()}}

    try:
        await _script(DEFAULT_CHANNELS[0], f"--paper-complete={arxiv_id}", timeout=60)
    except ScriptError as exc:
        # The note is on disk and `digest: full`; a second read would redo it.
        logger.warning("youtube-digest: --paper-complete failed for %s: %s", arxiv_id, exc)
    bits = [f"Paper {arxiv_id}: {title[:70]} — note written"]
    if unexpected:
        bits.append(f"{len(unexpected)} unexpected vault write(s)")
    return {"status": "success", "summary": _one_line(", ".join(bits)),
            "artifact_path": str(note_path), "response": text,
            "meta": {**run_meta, "note": str(note_path), "text_source": meta.get("text_source"),
                     "text_words": meta.get("text_words")}}


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


def _one_line(value: Any) -> str:
    """Collapse a value that may span lines into the one line a summary is.

    A failed fetch hands this source the fetch script's whole stderr, and every
    one of the six youtube-digest rows that hit the 500-character cap is exactly that: `ai-engineer g0vqT_wZtXA: Could not fetch
    transcript: Primary: Traceback (most recent call last):\\n  File "<string>",
    line 5…` — a stored run record that is seven physical lines, which no dashboard
    cell, fleet alert, or one-line-per-run `grep` renders as one record.
    The informative clause is already at the head ("Could not fetch
    transcript"), so this keeps the words and drops the layout: every run of
    whitespace, newline included, becomes one space.

    Deliberately applied here, by the source that phrases the summary, and not
    inside `queue._insert_run`. Length is the column's business and belongs at
    the insert; which words a source chooses, and whether it says them over one
    line, is the source's. Flattening at the insert would silently rewrite
    another source's summary too, and a source that wants a multi-line record
    should have to ask for it.
    """
    return " ".join(str(value).split())


async def _fail(channel: str, video_id: str, why: str) -> None:
    try:
        await _script(channel, f"--fail={video_id}", f"--reason={why[:400]}", timeout=60)
    except ScriptError as exc:
        logger.warning("youtube-digest: could not record failure for %s %s: %s", channel, video_id, exc)


async def execute(item: QueueItem) -> dict[str, Any]:
    from workers.sources import get_sources_config

    if item.kind == PAPER_KIND:
        return await _execute_paper(item)
    payload = item.payload or {}
    channel = str(payload.get("channel") or "")
    video_id = str(payload.get("video_id") or "")
    if not channel or not video_id:
        return {"status": "failed", "summary": "queue item carries no channel/video_id"}
    src_cfg = get_sources_config().get(NAME, {}) or {}
    base_meta = {"channel": channel, "video_id": video_id}

    # 1. The bundle. A transcript that cannot be fetched is recorded by the
    #    script itself (`--fetch` marks the attempt), so an `ok: False` needs
    #    nothing here but a report.
    #
    #    A *crash* is the other case, and it must be counted here: the script
    #    exited before it could mark anything, so the row is still `pending`
    #    and `pending_entries` offers it again on the next tick — forever, at
    #    one wasted run per tick per poisoned video, since nothing about the
    #    next attempt differs from this one. Counting it puts the row behind
    #    `_is_retry_eligible`'s 12 attempts at 15-minute spacing, which is
    #    the bound that turns a permanent crash into a video that gives up
    #    rather than a queue head that never moves.
    try:
        fetched = await _script(channel, f"--fetch={video_id}",
                                timeout=float(src_cfg.get("fetch_timeout_seconds", 420)))
    except ScriptError as exc:
        await _fail(channel, video_id, f"fetch crashed: {exc}")
        return {"status": "failed",
                "summary": f"{channel} {video_id}: fetch crashed: {_one_line(exc)}",
                "meta": {**base_meta, "fetch_crashed": True}}
    if not fetched.get("ok"):
        why = str(fetched.get("error") or "fetch failed")
        return {"status": "failed", "summary": f"{channel} {video_id}: {_one_line(why)}",
                "meta": {**base_meta, "fetch_error": why,
                         "failure_count": fetched.get("failure_count")}}
    meta = fetched["meta"]
    note_path = Path(meta["target_note"])
    title = str(meta.get("title") or video_id)

    # 2. The session.
    tracked = await asyncio.to_thread(tracked_items)
    try:
        prompt = await asyncio.to_thread(build_prompt, meta, tracked)
        prompt = prompt + _stage_a_footer(note_path)
    except SkillProtocolMissing as exc:
        # The protocol lives in the vault skill now (#737), so a deleted or
        # emptied skill leaves this dispatch holding a bundle and no
        # instructions. Fail by name instead: the row is not marked failed in
        # `seen.json`, because this is the deploy's fault and not the video's,
        # and the entry is offered again once the skill is back.
        logger.error("youtube-digest: %s", exc)
        return {"status": "failed", "summary": f"{channel} {video_id}: {_one_line(exc)}",
                "meta": {**base_meta, "skill_missing": True}}
    vault_before = await asyncio.to_thread(_vault_dirty_paths)
    want_structured = bool(payload.get("structured_verdict", True))
    try:
        run = await run_prompt_in_session(
            prompt, title=f"{meta.get('channel_name', channel)}: {title[:52]}",
            source=NAME,
            max_turns=int(payload.get("max_turns") or src_cfg.get("max_turns", 40)),
            priority=1,
            # Stage A of #2932: the digest's own deny plus every durable writer.
            extra_disallowed=list(TURN_ENVELOPES["stage-a-reader"]),
            final_schema=RESULT_SCHEMA if want_structured else None,
            final_schema_prompt=(
                "Restate the RESULT block above as a single JSON object "
                "matching the schema. Same outcome, same verdict, same filed id — "
                "this is a transcription, not a re-decision."
            ))
    except DrainActive as exc:
        # Not the video's fault: the entry stays `fetched` and is re-offered.
        return {"status": "skipped", "summary": f"landing in progress: {_one_line(exc)}",
                "meta": {**base_meta, "drain_active": True}}
    except TurnTimeout as exc:
        await _fail(channel, video_id, f"turn timeout: {exc}")
        return {"status": "failed", "summary": f"{channel} {video_id}: {_one_line(exc)}",
                "meta": {**base_meta, "turn_timeout": True}}

    session_id = run.get("session_id")
    text = run.get("text") or ""
    stop_reason = run.get("stop_reason")
    parsed = parse_verdict(text, run.get("structured"))
    structured_error = str(run.get("structured_error") or "")
    # #1709: this source keeps `fact_add` too (its deny list names only the
    # browser and registry writers), and a digest turn reads a page's own prose.
    # Its RESULT block claims no fact count, so there is nothing to disagree
    # with — recording `facts_written` alone is what turns a fact a page talked
    # the turn into from an invisible row into one named in `runs.meta_json`.
    verdict_meta = {"verdict_source": parsed["source"] if parsed else "none",
                    "structured_error": structured_error,
                    **await asyncio.to_thread(reconcile_fact_writes, session_id, "")}
    # ── 2b. The handoff and stage B (#2932) ───────────────────────────────────
    # Stage A had no write tool, so the document it was asked to emit travels as a
    # named artifact and a second turn lands it. The order below is the point: the
    # artifact is written, then RE-READ and re-validated, and only then is the writer
    # dispatched — so a refusal costs one session, not a bad file. Everything after
    # this block is the pre-split code unchanged: `--mark-note`, `_note_is_real`,
    # `_record_published` and the RESULT-block reconciliation now judge stage B's
    # output, which is why the pilot must not need a new on-disk check.
    #
    # Only a turn that CLAIMS it wrote the note gets a writer stage. A `kept` or
    # `failed` turn falls through to the disk checks below exactly as it did before.
    stage_b_meta: dict[str, Any] = {}
    if parsed and parsed.get("result") == "written":
        handoff_path, refusal = await asyncio.to_thread(
            write_handoff, video_id,
            _stage_b_payload(meta, note_path, _note_document(text),
                             str(parsed.get("verdict") or ""),
                             parsed.get("relevance")))
        stage_b_meta = {"handoff_path": str(handoff_path)}
        if not refusal:
            _payload, refusal = await asyncio.to_thread(read_handoff, video_id)
        if refusal:
            # The shape of the pre-split message is kept (`without a note` at the
            # path), because that is what a reader of `runs.summary` has been
            # matching for a year; what it now adds is WHY, which used to be
            # unknowable — the turn said `written` and nothing was on disk.
            why = (f"turn ended ({run.get('stop_reason')}, {run.get('num_turns')} "
                   f"iterations) without a note at {note_path.name} "
                   f"(handoff refused: {refusal})")
            await _fail(channel, video_id, why)
            return {"status": "failed",
                    "summary": f"{channel} {video_id}: {_one_line(why)}",
                    "response": text,
                    # `stop_reason` and `empty_response` ride along because the
                    # pre-split no-note failure carried them, and the readers of
                    # `runs.meta_json` — the failure-ledger rollup above all — group
                    # this row with that one.
                    "meta": {**base_meta, "session_id": session_id,
                             "stop_reason": stop_reason,
                             "empty_response": not text.strip(),
                             "handoff_refusal": refusal, **stage_b_meta}}
        try:
            writer_run = await _run_writer_stage(
                _payload, handoff_path,
                max_turns=int(payload.get("max_turns") or src_cfg.get("max_turns", 40)),
                title=f"{meta.get('channel_name', channel)}: {title[:52]}", priority=1,
                item_id=item.id)
        except DrainActive as exc:
            # The writer never got to run: the artifact is on disk and the row is not
            # counted, so the retry lands the same document rather than re-reading it.
            return {"status": "skipped",
                    "summary": f"landing in progress: {_one_line(exc)}",
                    "meta": {**base_meta, "drain_active": True, **stage_b_meta}}
        except Exception as exc:  # noqa: BLE001 — a crashed writer is a failed run
            logger.exception("youtube-digest: stage B crashed for %s %s", channel, video_id)
            await _fail(channel, video_id, f"writer stage crashed: {exc}")
            return {"status": "failed",
                    "summary": f"{channel} {video_id}: writer stage crashed: "
                               f"{_one_line(exc)}",
                    "meta": {**base_meta, "writer_crashed": True, **stage_b_meta}}
        stage_b_meta.update({"stages": 2, "writer_session_id": writer_run.get("session_id"),
                             "writer_turns": writer_run.get("num_turns"),
                             "writer_stop_reason": writer_run.get("stop_reason")})

    on_disk = await asyncio.to_thread(_note_is_real, note_path, video_id)
    # Measured across BOTH stages now: a vault path that moved during either
    # turn is unexpected, and stage B is exactly the stage whose writes must
    # be the note and nothing else.
    unexpected = await asyncio.to_thread(_unexpected_vault_writes, vault_before)

    # 3a. An infra-shaped turn: no text and no stop reason means the harness
    #     never got a completion (engine unreachable, aggregator restarting).
    #     Not the video's fault, so it is not counted against its retries: this
    #     branch calls no `--fail`, `seen.json` is untouched, the row stays
    #     `fetched` and the bundle is kept. On 2026-09-09 the primary was down
    #     for a launcher change and six videos went through this path in five
    #     seconds each, and every one of them was right to be tried again.
    #
    #     What it did not have was a spacing. `fetched` is a queueable status and
    #     `_is_retry_eligible` gates only `failed` rows, so "tried again" meant a
    #     new row on every 300 s tick for as long as the engine stayed away, with
    #     no attempt counted anywhere (#1714). `defer_seconds` is the bound: the
    #     pool parks THIS row behind `not_before` for `INFRA_DEFER_SECONDS` (900 s)
    #     instead of completing it, and keeps its dedup key, so the next tick's
    #     `enqueue` coalesces against it rather than minting a fresh row. The
    #     transient case above pays 15 minutes; the outage case stops multiplying.
    if not text.strip() and stop_reason is None:
        errs = [str(e)[:160] for e in (run.get("errors") or [])[:2]]
        why = f"turn produced nothing ({'; '.join(errs) or 'no error reported'})"
        logger.warning("youtube-digest: %s %s: %s — left fetched, retried in %ds",
                       channel, video_id, why, INFRA_DEFER_SECONDS)
        return {"status": "failed", "summary": f"{channel} {video_id}: {_one_line(why)}",
                "defer_seconds": INFRA_DEFER_SECONDS,
                "meta": {**base_meta, "session_id": session_id, "infra": True,
                         "empty_response": True, "unexpected_vault_writes": unexpected}}

    # 3b. Disk decides — with one refinement. A note that did not exist before
    #     and does now is proof the turn wrote it. A note that *pre-existed*
    #     (AI Engineer's back catalogue) proves nothing about this turn, so
    #     for a refresh the evidence is the RESULT block, or at least a clean
    #     stop with text. Without that rule a failed turn over an old note was
    #     recorded as completed "no verdict" — four of them on 2026-09-09.
    #     Once the finalizer has run, that fallback is gone: it runs only on a
    #     clean stop, so "a clean stop with text" would pass every turn it ran
    #     on, including one whose verdict it could not read — a verdict with a
    #     known result is required instead, from either path.
    finalizer_ran = want_structured and (
        run.get("structured") is not None
        or bool(structured_error and "skipped" not in structured_error))
    has_verdict = parsed is not None and parsed.get("result") in _RESULTS
    if finalizer_ran:
        turn_ran = has_verdict
    else:
        turn_ran = parsed is not None or (stop_reason in ("stop", "end_turn") and bool(text.strip()))
    if not on_disk or (meta.get("existing_note") and not turn_ran):
        what = "without a note" if not on_disk else "without a RESULT block over a pre-existing note"
        why = f"turn ended ({stop_reason}, {run.get('num_turns')} iterations) {what} at {note_path.name}"
        await _fail(channel, video_id, why)
        return {"status": "failed", "summary": f"{channel} {video_id}: {_one_line(why)}",
                "response": text,
                "meta": {**base_meta, "session_id": session_id,
                         "stop_reason": stop_reason,
                         "empty_response": not text.strip(),
                         "unexpected_vault_writes": unexpected, **verdict_meta}}

    eval_result = _eval_record(parsed, session_id)
    if eval_result.get("filed"):
        ok = await asyncio.to_thread(_filed_item_exists, int(eval_result["filed"]), video_id)
        if not ok:
            eval_result["filed_unverified"] = eval_result.pop("filed")
            logger.warning("youtube-digest: %s %s claimed FILED #%s but no such item is on disk",
                           channel, video_id, eval_result["filed_unverified"])

    # 4. Record, then refresh the report note (best effort).
    try:
        await _script(channel, f"--complete={video_id}", f"--note={note_path}",
                      f"--eval-json={json.dumps(eval_result)}", timeout=60)
    except ScriptError as exc:
        # The note exists and the verdict is in this run record; a second
        # attempt would redo the whole session, so this is a warning.
        logger.warning("youtube-digest: --complete failed for %s %s: %s", channel, video_id, exc)
    try:
        await _script(channel, "--eval-report", timeout=60)
    except ScriptError as exc:
        logger.warning("youtube-digest: --eval-report failed for %s: %s", channel, exc)

    # 5. The papers the turn named. After `--complete`, because the registry
    #    links each paper back to the note that call records. Best effort: an
    #    arXiv outage costs these papers their notes, never the video its run.
    papers_meta: dict[str, Any] = {}
    refs = (parsed or {}).get("papers") or []
    if refs and _papers_cfg(src_cfg)["enabled"]:
        try:
            out = await _script(channel, f"--paper-register={video_id}",
                                f"--paper-refs={json.dumps(refs)}", timeout=180)
            papers_meta = {"papers_registered": [r.get("arxiv_id") for r in out.get("registered") or []],
                           "papers_unresolved": list(out.get("unresolved") or [])}
        except ScriptError as exc:
            logger.warning("youtube-digest: --paper-register failed for %s %s: %s", channel, video_id, exc)
            papers_meta = {"papers_register_error": _one_line(exc)[:300]}

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
    if stage_b_meta.get("stages") == 2:
        # Says out loud, in the row a human reads, that the file on disk came from
        # the writer stage rather than from the turn that read the transcript.
        bits.append("note landed by the writer stage")
    if parsed and parsed.get("result") == "kept":
        bits.append("note kept")
    if papers_meta.get("papers_registered"):
        bits.append(f"{len(papers_meta['papers_registered'])} paper(s) registered")
    if unexpected:
        bits.append(f"{len(unexpected)} unexpected vault write(s)")
    return {
        "status": "success",
        # A video title is model-supplied metadata and can carry a newline, and
        # no source-side `[:500]` survives in this file: `queue._insert_run`
        # owns the length, on a word boundary, with the marker.
        "summary": _one_line(", ".join(bits)),
        "artifact_path": str(note_path),
        "response": text,
        "meta": {**base_meta, "session_id": session_id, "note": str(note_path),
                 "stop_reason": run.get("stop_reason"), "num_turns": run.get("num_turns"),
                 # `stages: 2`, the artifact path and stage B's turn count. Readable
                 # after the fact, which is what owed clause 1 needs: "zero production
                 # refusals on the writer stage" is a query over these keys.
                 "eval": eval_result, "unexpected_vault_writes": unexpected,
                 "parsed": parsed is not None, **verdict_meta, **papers_meta,
                 **stage_b_meta},
    }


# ══════════════════════════════════════════════════════════════════════════════
# The two-stage split (#2932): a reader stage with no durable write, a writer
# stage with no ingest reach.
#
# #2269 gave this source a capability allow-list, which narrows WHICH tools a
# source can reach inside ONE turn, and its own residual-risk paragraph named what
# that leaves behind: "an untrusted-ingest turn that legitimately must write durably
# (a digest job writes a note) can still be steered into a bad durable write."
# Measured at triage 2026-10-17, this source's compiled envelope (n=23) held
# `http_fetch` and `http_search` beside `Write`, `vault_write`, `backlog_write_task`,
# `fact_add`, `fact_relate` and `fact_invalidate` — and every one of those writers is
# tier-1, which is why #2269's roster node, keyed on `durable_external()` =
# TIER2 ∪ TIER3, structurally could not see the co-presence.
#
# So the change here is topological, not policy. Stage A reads the third-party
# transcript with `http_*` reachable and NO durable writer; stage B holds the note
# write, has no fetch/egress verb, and is handed a named, schema-validated artifact
# instead of the content. What becomes true is a property about the topology: no
# dispatch context pairs inbound untrusted text with both a write verb and an egress
# read. It costs no model call and no GPU — the gate is a declared deny set and a
# schema check.
#
# What is NOT bought, said once so nothing downstream over-claims it: stage B lands
# text stage A drafted while under injection influence. What changes is that THE
# WRITE TARGET AND THE VERB ARE CHOSEN BY A CONTEXT THAT NEVER SAW THE PAYLOAD.
# Injection resistance per se is unchanged — `run_injection_canary` (#590) and the
# outbound content gate (#1136) keep that job. #2269's sentence is the line this
# inherits, and this does not close it either.
# ══════════════════════════════════════════════════════════════════════════════

#: Stage A's deny, on top of the digest's own `DISALLOWED`.
#:
#: Derived from the turn's observed calls rather than from what the job description
#: implies — #2269's `session-distill` finding (344 of 560 calls were `Bash`, so a
#: derived set that omitted it would have clipped the job) is the precedent. Stage A's
#: real work is `Read`/`Grep`/`Glob` on the bundle, the measurement scripts,
#: `http_fetch` on the official page, `Task` for the PAPERS handoff, and the finalizer:
#: every one of those stays reachable. What goes is the durable half — the note write,
#: the vault, the backlog, the fact store, memory — plus `Scratchpad`, which is
#: `side_effecting()` on both classifiers (#2269 denied it for the same reason in
#: `SOURCE_CAPABILITIES`) and which this stage has no use for: its whole output travels
#: as its reply.
_STAGE_A_EXTRA_DENY = frozenset({
    "Write", "Edit", "vault_write", "vault_overview", "backlog_write_task",
    "backlog_boards", "backlog_tasks", "backlog_get_task",
    "fact_add", "fact_relate", "fact_invalidate", "memory_add", "memory_replace",
    "memory_remove", "tasks_create", "tasks_update", "calendar_create",
    "calendar_update_event", "calendar_delete_event", "contacts_create",
    "contacts_update", "contacts_delete", "autonomy_write_task",
    "research_propose", "Scratchpad",
})

#: Every third-party read / egress verb, named once.
#:
#: This is the half of the #2932 pair that a context holding untrusted content must
#: not share with a durable write, so it is one set rather than a copy per turn shape:
#: stage B denies it because stage B is the writer, and the PAPER turn denies it
#: because the paper turn reads a whole third-party document beside its `Write`. By
#: the same triage that counted the video envelope at n=23, `PAPER_DISALLOWED` leaves
#: `http_fetch`/`http_search`/`browser_navigate` reachable beside `Write` and
#: `vault_write` in the paper turn (`_execute_paper`, `extra_disallowed=list(
#: PAPER_DISALLOWED)`), so measuring only the two video stages would have published a
#: clean row for a source whose third shape still pairs the halves.
#:
#: Nothing here costs the paper job a tool it uses: `paper-digest` says it in its own
#: body — "The producer already fetched the paper onto disk … Do not search for the
#: paper and do not fetch it" (`~/obsidian/skills/paper-digest/SKILL.md:20-21`) — and
#: the fetch is a script call (`--paper-fetch`), not a turn tool.
_EGRESS_DENY = frozenset({
    "http_search", "http_fetch", "http_request",
    "browser_navigate", "browser_snapshot", "browser_scroll", "browser_press",
    "browser_select", "browser_tabs", "browser_wait", "browser_cookies",
    "browser_evaluate", "browser_click", "browser_fill", "browser_drag",
    "browser_screenshot",
})

#: Stage B's deny: every third-party read/egress verb, and every durable writer that
#: is not the note itself.
#:
#: `_EGRESS_DENY` is the load-bearing half — `http_fetch` beside `Write` in one
#: envelope is the combination this item exists to remove. The rest is the same
#: one-way rule: a stage whose only job is to land a validated document has no use for
#: the fact store, the backlog or memory, and every name left reachable is a name by
#: which that context could still be steered. `Bash`, `Edit` and `Task` are already in
#: `DISALLOWED` for this source, so they need no restating here.
_STAGE_B_DENY = _EGRESS_DENY | frozenset({
    "backlog_write_task", "backlog_boards", "backlog_tasks", "backlog_get_task",
    "fact_add", "fact_relate", "fact_invalidate", "memory_add", "memory_replace",
    "memory_remove", "tasks_create", "tasks_update", "calendar_create",
    "calendar_update_event", "calendar_delete_event", "contacts_create",
    "contacts_update", "contacts_delete", "autonomy_write_task",
    "research_propose", "Scratchpad",
})

#: Every turn shape this module dispatches, by the deny set its own call site passes as
#: `extra_disallowed`.
#:
#: This is the census's denominator, so it is a property rather than a convenience:
#: `tests/test_worker_capability_allowlist.py` compiles one row per entry, and its
#: AST node refuses a `extra_disallowed=` site in this module that is not a lookup
#: here. Without that, the table could record a split source as fixed while a third
#: shape it never measured still paired the halves — the exact way the first version
#: of this file got it wrong with the paper turn.
TURN_ENVELOPES: dict[str, tuple[str, ...]] = {
    #: Stage A of the video path: reads the transcript, keeps egress, no durable writer.
    "stage-a-reader": tuple(sorted(set(DISALLOWED) | _STAGE_A_EXTRA_DENY)),
    #: Stage B: lands the validated note, no egress verb at all.
    "stage-b-writer": tuple(sorted(set(DISALLOWED) | _STAGE_B_DENY)),
    #: The PAPERS child: one whole third-party document beside one note write, so the
    #: egress half of the pair goes. Its durable writes stay — it legitimately writes.
    "paper-note-turn": tuple(sorted(set(PAPER_DISALLOWED) | _EGRESS_DENY)),
}

#: What the two stages are called in every refusal, log line and census row. The
#: queue item's `source` stays `youtube-digest`: this splits ONE job into two turns, it
#: does not register a worker, and inventing a source name here would put a name in the
#: roster that `worker_source_names()` cannot resolve.
STAGE_A_NAME = "youtube-digest/reader"
STAGE_B_NAME = "youtube-digest/writer"

#: Where stage A's handed-over document goes, and what it is called. Under the
#: pipeline's pending-research root, in a directory per video — deliberately NOT the
#: target note path, because stage A must not be able to write there, and a handoff
#: that sits at the final destination is a write with extra steps.
HANDOFF_FILENAME = "stage-a.json"

#: The first line of every stage B prompt. One marker so a test, a refusal log and a
#: run record can name the writer stage's turn without re-deriving it from the envelope.
HANDOFF_MARK = "# Digest note write (stage B)"

#: The declared payload: everything stage B is allowed to see, and nothing else. Every
#: field is required; a missing one is a schema failure, not a blank to write around.
#: `relevance` is the one integer, and -1 is a legal value meaning "the RESULT block
#: stated no number" — the same sentinel `_eval_record` uses, not a fabricated score.
HANDOFF_FIELDS: tuple[str, ...] = ("video_id", "title", "url", "channel_key",
                                   "published", "target_note", "note_document",
                                   "verdict", "open_questions_path")
HANDOFF_INT_FIELDS: tuple[str, ...] = ("relevance",)
HANDOFF_SCHEMA_VERSION = 1

#: Stage A's note body travels as its last section, fenced, because `_extract_section`
#: stops at the next `## ` and the note has headings of its own. Same fenced-JSON
#: convention as `META:` above: the fence is what makes a heading inside the payload
#: harmless.
_NOTE_FENCE_RE = re.compile(
    r"(?m)^##\s*NOTE_DOCUMENT\b[^\n]*\n+```[^\n]*\n(.*?)```", re.DOTALL)


def reader_run_options(cfg: dict | None = None, max_turns: int | None = None):
    """Stage A's turn options: the digest's own deny plus every durable writer.

    The same `_worker_run_options` route as the pre-split turn and the same `source`
    this source has always named itself by, so the split tightens one source's
    declaration instead of inventing a capability entry nobody declared.
    """
    cfg = cfg or {}
    return _worker_run_options(
        int(max_turns or cfg.get("max_turns") or 40), source=NAME,
        extra_disallowed=list(TURN_ENVELOPES["stage-a-reader"]))


def writer_run_options(cfg: dict | None = None, max_turns: int | None = None):
    """Stage B's turn options: every egress verb denied, `Write` left reachable."""
    cfg = cfg or {}
    return _worker_run_options(
        int(max_turns or cfg.get("max_turns") or 40), source=NAME,
        extra_disallowed=list(TURN_ENVELOPES["stage-b-writer"]))


def paper_run_options(cfg: dict | None = None, max_turns: int | None = None):
    """The PAPERS child's turn options: its own deny plus every egress verb."""
    cfg = cfg or {}
    return _worker_run_options(
        int(max_turns or cfg.get("max_turns") or 40), source=NAME,
        extra_disallowed=list(TURN_ENVELOPES["paper-note-turn"]))


def handoff_dir(video_id: str) -> Path:
    """Where this video's stage-A document lives."""
    return VAULT_PENDING_RESEARCH_DIR / NAME / "stage-a" / str(video_id)


def _stage_b_payload(meta: dict, note_path: Path, document: str, verdict: str,
                     relevance: int | None) -> dict:
    """The declared payload from the bundle's own metadata — never from the content."""
    return {
        "video_id": str(meta.get("video_id") or ""),
        "title": str(meta.get("title") or ""),
        "url": str(meta.get("url") or ""),
        "channel_key": str(meta.get("channel_key") or ""),
        "published": str(meta.get("published") or ""),
        "target_note": str(note_path),
        "note_document": document,
        "verdict": str(verdict or ""),
        "open_questions_path": str(handoff_dir(str(meta.get("video_id") or ""))
                                   / "open-questions.json"),
        "relevance": relevance if isinstance(relevance, int) and not isinstance(
            relevance, bool) else -1,
    }


def _handoff_refusal(payload: dict, video_id: str,
                     min_bytes: int = _MIN_NOTE_BYTES) -> str:
    """Why this payload may NOT be written, or "" when it may.

    Every check is a fact the writer stage can verify from the document itself, because
    that is the value of the handoff: stage B does not trust stage A's prose, it
    re-reads the artifact. `_MIN_NOTE_BYTES` is the source's own on-disk floor, so a
    stub cannot pass the gate here and then fail the census there — and a document
    whose `video_id` is not THIS bundle's is the one wrong-write shape that matters
    most: the item's whole claim is that the write target is chosen without the
    payload in the room.
    """
    for field in HANDOFF_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"handoff_missing_field:{field}"
    for field in HANDOFF_INT_FIELDS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            return f"handoff_bad_type:{field}"
        if field == "relevance" and not -1 <= value <= 100:
            return f"handoff_out_of_range:{field}"
    if payload["video_id"] != str(video_id):
        return f"handoff_video_id_mismatch:{payload['video_id']}"
    document = payload["note_document"]
    if not document.lstrip().startswith("---"):
        return "handoff_no_front_matter"
    if len(document.encode("utf-8")) < min_bytes:
        return f"handoff_note_too_short:{len(document.encode('utf-8'))}"
    return ""


def write_handoff(video_id: str, payload: dict) -> tuple[Path, str]:
    """Stage A's output → the named artifact. Returns (path, refusal); a refusal wins.

    Validated BEFORE the write, so a document that could never be landed is not left on
    disk for a later run to mistake for a handoff.
    """
    path = handoff_dir(video_id) / HANDOFF_FILENAME
    refusal = _handoff_refusal(payload, video_id)
    if refusal:
        return path, refusal
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": HANDOFF_SCHEMA_VERSION, "source": NAME,
                                "stage": STAGE_A_NAME, "payload": payload},
                               ensure_ascii=False), encoding="utf-8")
    return path, ""


def read_handoff(video_id: str) -> tuple[dict, str]:
    """Read and re-validate stage B's input. Returns (payload, refusal).

    The three shapes are the three ways a producer can lie — it wrote nothing, it wrote
    something unparseable, it wrote JSON that is not the declared payload — and each
    gets its own token so the run record says which. None of them is "write something
    anyway": that is the failure mode this whole seam exists to make impossible, and the
    reason this returns a refusal rather than a best-effort dict.
    """
    path = handoff_dir(video_id) / HANDOFF_FILENAME
    if not path.is_file():
        return {}, f"handoff_absent:{path}"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — every parse failure is the same refusal
        return {}, f"handoff_unparseable:{type(exc).__name__}"
    if not isinstance(obj, dict) or not isinstance(obj.get("payload"), dict):
        return {}, "handoff_not_an_object"
    payload = obj["payload"]
    refusal = _handoff_refusal(payload, video_id)
    if refusal:
        return {}, refusal
    return payload, ""


def _note_document(text: str) -> str:
    """The note body stage A was asked to hand over, or "" when it did not emit one."""
    m = _NOTE_FENCE_RE.search(text or "")
    return m.group(1).rstrip() if m else ""


def _stage_a_footer(note_path: Path) -> str:
    """What stage A is told instead of "write the note": emit it, do not write it.

    Appended to the digest prompt after the skill's own instructions, so the reader
    stage's deny cannot surprise it: the skill body still names `Write`, and the footer
    says in one place that the tool is not reachable here and what to do instead.
    """
    return ("\n\n---\n\n"
            "## NOTE_DOCUMENT\n\n"
            "You have NO file-write tool on this turn, and you must not try to use one: "
            "this is the READER half of a two-stage digest (#2932). A separate writer "
            "turn, with no web access and no view of this transcript, lands the file. So "
            "emit the complete note as your last section, inside a fenced code block, "
            "verbatim — front matter included, the `video_id:` line matching this bundle "
            f"exactly, `## Open Questions` last inside it, and nothing after the fence. "
            f"It is the same document you would have written to {note_path}. Anything you "
            "leave out of that block is lost: no one can ask you again.\n")


def _stage_b_prompt(payload: dict, path: Path) -> str:
    """Stage B's whole prompt: the named artifact, the declared fields, the rules.

    Deliberately short and deliberately not the digest. No `transcript_path`, no
    `text_path`, no skill body and no channel profile — the only text this context can
    put into a file is text that already survived validation.
    """
    return (
        f"{HANDOFF_MARK}\n\n"
        "You are the WRITER half of a two-stage YouTube digest (#2932). You have no web "
        "access on this turn and you must not ask for any: what you are about to write "
        "was read from the artifact named below, not fetched by you, and you cannot "
        "check it against the source. That is the point — the turn that read the "
        "third-party transcript has no write tool, and this turn has no reader tools.\n\n"
        f"Artifact: {path}\n\n"
        "Declared payload (already schema-validated: fields present, front matter, "
        "minimum size, video_id matches this bundle):\n"
        + "".join(f"- {key}: {payload[key]}\n" for key in
                  ("video_id", "title", "url", "channel_key", "published", "verdict",
                   "relevance", "open_questions_path", "target_note")) +
        "\nDo exactly this:\n"
        f"1. Read {path} and take `payload.note_document` from it.\n"
        f"2. Write it verbatim to {payload['target_note']} with the Write tool. Do not "
        "rewrite, shorten, translate or improve it, and do not add a line of your own.\n"
        "3. Reply with the single word: written\n\n"
        "If that file is missing, unparseable, or has no `note_document`, write NOTHING "
        "and do not reconstruct a note from the fields above — reply `refused` and name "
        "the reason. A missing note is a failure someone can re-run; a note assembled "
        "from a summary is a file that looks like a digest and is not one.\n")


async def _run_writer_stage(payload: dict, path: Path, *, max_turns: int,
                            title: str, priority: int, item_id: int | None = None,
                            finalizer=None):
    """Dispatch stage B on its OWN fresh session and return the turn dict.

    A fresh session, not stage A's: a session that carried the reader turn would carry
    the transcript into the writing context and undo the split under a new label.
    """
    return await run_prompt_in_session(
        _stage_b_prompt(payload, path), title=f"{title} [stage B]", source=NAME,
        max_turns=max_turns, priority=priority,
        # The deny travels as `extra_disallowed`, because that is the only route this
        # helper has into `_pre_dispatch` — it takes no `run_options`, and passing one
        # would raise `TypeError` on the first real digest (invisible to every fake in
        # the test file, which accepts `**kwargs`). `writer_run_options` compiles the
        # same union the census measures, and the node in
        # `tests/test_youtube_digest_source.py` pins that the two agree.
        extra_disallowed=list(TURN_ENVELOPES["stage-b-writer"]))
