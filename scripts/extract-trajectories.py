#!/usr/bin/env python3
"""
extract-trajectories.py — Parse session files into structured trajectory logs.

Reads Lloyd session JSON files from ~/lloyd-data/sessions/*.json

Use --agent worker to process only autonomy sessions (autonomy_*.json)
Use --agent main  to process only interactive sessions (non-autonomy *.json)

Output: ~/lloyd-data/_pipeline/trajectories/YYYY-MM-DD.jsonl
State:  ~/lloyd-data/_pipeline/trajectories/.watermark.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Daily notes are dated by local time (America/Los_Angeles, PST/PDT).
# Bucket trajectory output by local date so {date}.jsonl aligns with
# memory/learnings/{date}.md — UTC bucketing misfiles 17:00-24:00 PDT
# sessions one day late (flagged 2026-08-19/20, fixed 2026-08-21).
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────
#
# LLOYD_SESSIONS, OUTPUT_DIR and WATERMARK_PATH are bound from `app.paths` below
# `_import_root_on_path()`, which has to run before any `app.*` import.

# ── Session class (#493) ─────────────────────────────────────────────────────
#
# The frequency gate qualifies a pattern on *distinct sessions*, so a corpus that
# is mostly the loop's own traffic ranks the loop. Machine share of the corpus,
# measured on the `platform` field in ~/lloyd/sessions/*.json joined onto
# _pipeline/trajectories/*.jsonl: 09-02 0/6, 09-08 97/110, 09-09 338/348,
# 09-11 305/314, 09-12 15/15 — 938 of 1083 (86.6%) across the window, and
# 10,611 of 11,813 `Sessions Affected` bullets in the 2026-09-12 candidates.
#
# The classifier that should have caught this keyed on the filename
# (`path.stem.startswith("autonomy_")`) and no live session file matches it: 0 of
# the 1,686 files in `~/lloyd/sessions/` on 2026-09-14 (0 of 1,331 at triage),
# because the loop renamed itself around 09-09 (`youtubed_*`, `autocode_*`,
# `autotriage_*`, `benchmine_*`). Every row came out `agent_id: "lloyd"` — 1,306 of
# 1,306 rows across the seven newest buckets on 2026-09-14 (1,014 of 1,014 at
# triage) — so class was unknowable downstream. Re-classifying the whole corpus
# through the fields below, measured the same day: 1,385 of 1,411 rows (98.2%) are
# machine sessions and 26 are interactive.
#
# Class is therefore read from fields every session JSON already carries.
INTERACTIVE_CLASS = "interactive"   # a chat a person typed into

# ── Reclassified by backlog #1143 (Alan's ruling, 2026-09-14) ────────────────
#
# #493 keyed `interactive` on `platform == "mission-control"` AND `inner_voice`
# falsy, which had the flag's meaning backwards. `inner_voice: true` is the switch
# that turns the observer ON for a human's chat — set at creation by
# `web/src/components/pages/InnerVoicePage.tsx:146`, the sidebar chat and the
# Chrome extension — not a mark of an observer-only turn. So the rule excluded
# every chat the observer was enabled for and kept what the flag happened to miss:
# measured over the 14 days to 2026-09-14, 79 excluded `inner-voice` sessions are
# real conversations of up to 26 user turns while 42 of the 42 kept
# `interactive` sessions are test ids (`v2-*`, `dbg*`, `soak*`, `*_selftest_*`,
# sandbox four-part ids). The first post-landing nightly kept 4 of 1,471 sessions
# and mined nothing. The flag therefore plays no part in the class.
#
# Admission is `app.sessions_io.is_user_session` — the one definition of "a human
# is reading this session" that the ambient-delivery path already uses — and not a
# second private rule here, because a deny-list maintained in two places diverges:
# a platform added to `NON_USER_PLATFORMS` for the chat listings must not keep
# streaming into the mining corpus. Importing it is also why this module carries
# the class *sets* the miner filters on: the writer and the reader of the corpus
# then cannot drift, which is what #493's cross-file pin was holding shut.
#
# A human-platform session whose id is not the three-part chat shape is a script
# that drove it, whatever `platform` the creation endpoint was handed — that is
# exactly the 42 test ids above. `e2e-harness` is decided by platform first, so
# its harness smokes stay `smoke` and keep their own reporting bucket.
def _import_root_on_path() -> None:
    """Make `app.*` importable when this script is run by path.

    The nightly invokes `python3 ~/lloyd/scripts/extract-trajectories.py`, which
    puts `scripts/` on `sys.path` and not the repo root, so the plain
    `from app.sessions_io import ...` below would resolve only when pytest
    happens to have the checkout on the path. `scripts/backfill_session_titles.py`
    and `scripts/entity_resolution.py` do the same insertion for the same reason.
    """
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_import_root_on_path()

from app.sessions_io import is_user_session  # noqa: E402
from app.paths import PIPELINE_DIR, SESSIONS_DIR  # noqa: E402

LLOYD_SESSIONS = SESSIONS_DIR
OUTPUT_DIR = PIPELINE_DIR / "trajectories"
WATERMARK_PATH = OUTPUT_DIR / ".watermark.json"

#: A chat id is `<8 digits>_<6 digits>_<suffix>` — `20260912_101010_9f2a1c` for a
#: Mission Control chat, `..._iv0484` for one the inner voice was enabled on. The
#: same shape `app.sessions_io.is_background_session_name` discriminates on; here
#: it separates a conversation from a script that borrowed a human platform.
CHAT_SESSION_ID_RE = re.compile(r"^\d{8}_\d{6}_[A-Za-z0-9]+$")

# platform -> class. `browser` keeps its own label rather than folding into
# `interactive`: it is the Chrome extension's side panel
# (`chrome-extension/src/background/lloyd-client.ts:15`), a different surface with
# its own failure modes, so a candidate that only ever came from it stays
# attributable — and it is still a human session, so it is admitted (Alan's ruling:
# "kept by default, under their own `browser` label so they can still be reported
# separately").
SESSION_CLASS = {
    "mission-control": INTERACTIVE_CLASS,
    "worker": "worker",
    "autonomy": "autonomy",
    "browser": "browser",
    "e2e-harness": "smoke",
}
UNKNOWN_CLASS = "unknown"           # no `platform`, or one no deny-list mentions
TEST_CLASS = "test"                 # human platform, non-chat id: a script drove it

#: The classes a person's own work appears in — what the mining exclusion admits.
#: A new human-facing platform is one line in `SESSION_CLASS` plus one here, and
#: `mine-trajectories.py` changes in neither case.
HUMAN_CLASSES = frozenset({INTERACTIVE_CLASS, "browser"})
#: What the exclusion drops. Neither set may contain a class the other has;
#: `tests/test_trajectory_extraction.py` pins that.
MACHINE_CLASSES = frozenset({"worker", "autonomy", "smoke", TEST_CLASS})

# ── Sensitive data patterns ───────────────────────────────────────────────────

SENSITIVE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{10,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|apikey|secret|token|password|passwd|auth)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{36}", re.IGNORECASE),   # GitHub PAT
    re.compile(r"xoxb-[A-Za-z0-9\-]+", re.IGNORECASE),  # Slack bot token
]

MAX_STRING_LEN = 500
MAX_FILE_CONTENT_LEN = 200
MAX_ERROR_LEN = 200

# Keys whose values are likely to contain file content (truncate aggressively)
CONTENT_ARG_KEYS = {"content", "text", "body", "data", "message"}

# Keys whose value is a shell command. For these, truncation keeps the leading
# program token instead of dropping the whole string.
#
# The placeholder used to replace the entire value (`[truncated: 900 chars]`),
# and the miner groups on the first whitespace token of the value — so every
# over-long command in the corpus, whatever it did, landed in one bucket. It
# became the #2 skill candidate by occurrences (625, across 70 sessions) and an
# unsorted union of transcript extraction, awk, nvidia-smi and git calls, i.e. a
# description of this extractor rather than of any procedure. Backlog #391.
#
# The verb costs ~10 chars and is the only part of the string that says what the
# call was. Bound it: a first token can itself be enormous (an inline heredoc,
# a 2 KB path).
COMMAND_ARG_KEYS = {"command", "cmd"}
COMMAND_HEAD_KEEP = 60


# ── Error categorization ──────────────────────────────────────────────────────

ERROR_CATEGORIES = [
    ("permission",  re.compile(r"permission denied|access denied|forbidden|EPERM|EACCES", re.IGNORECASE)),
    ("not_found",   re.compile(r"file not found|no such file|not found|404|ENOENT", re.IGNORECASE)),
    ("timeout",     re.compile(r"timeout|timed out|ETIMEDOUT|deadline exceeded", re.IGNORECASE)),
    ("network",     re.compile(r"connection refused|ECONNREFUSED|DNS|ENOTFOUND|network|EHOSTUNREACH", re.IGNORECASE)),
    ("validation",  re.compile(r"invalid|malformed|parse error|syntax error|schema|validation", re.IGNORECASE)),
    ("resource",    re.compile(r"out of memory|disk full|quota|ENOMEM|ENOSPC|resource exhausted", re.IGNORECASE)),
]


# Failure *vocabulary* a tool result might contain. This is a descriptive
# signal only — it promotes nothing. It used to be the semantic error detector
# and set `is_error` on its own, which is what made the trajectory error signal
# unreadable: it matched the text a tool *returned*, not whether the call
# failed. Measured over the 2026-09-06→08 window, 196 of 234 flagged steps were
# flagged by these patterns alone and 188 of the 234 had nothing behind them
# (`_pipeline/skills/candidates/REVIEW-LOG.md`); a `Read/timeout` skill
# candidate was even filed for a tool with no timeout path, because the word
# came from the file being read. See backlog #389.
MENTION_ERROR_PATTERNS = [
    # Python exceptions
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"(?:Error|Exception|Warning):\s+\S", re.IGNORECASE),
    # Shell errors
    re.compile(r"(?:command not found|No such file or directory|Permission denied|Operation not permitted)", re.IGNORECASE),
    re.compile(r"bash: \S+: ", re.IGNORECASE),
    # Node/npm errors
    re.compile(r"(?:npm ERR!|SyntaxError:|ReferenceError:|TypeError:)", re.IGNORECASE),
    # Common failure keywords in result text (conservative — require context)
    re.compile(r"^FAILED\b", re.MULTILINE),
    re.compile(r"\bfatal error\b", re.IGNORECASE),
]

# Kept as an alias: `has_semantic_error` still answers "does this text use
# failure vocabulary", which is what the mention flag below is.
SEMANTIC_ERROR_PATTERNS = MENTION_ERROR_PATTERNS

# The Bash tool reports its exit state as a trailer, e.g. "...output\n\n[exit
# code: 1]". The trailer is the last thing in the body, so it is gone before
# `result_summary()` runs (MAX_ERROR_LEN keeps a 200-char *prefix*) — 44 of the
# 67 non-zero exits in the 09-06→08 window are invisible in the persisted
# preview (backlog #492). It is therefore read from the full result here and
# persisted as `exit_code`. A negative value is a signal death (SIGTERM = -15),
# which is a failure too.
EXIT_CODE_RE = re.compile(r"\[exit code:\s*(-?\d+)\]\s*$")

# The exit trailer is tool output, not data; it is not failure vocabulary.
MENTION_EXCLUDE_EXIT_TRAILER = re.compile(r"\s*\[exit code:\s*-?\d+\]\s*$")

# Content-returning tools. Their result is the thing asked for, so the words in
# it carry zero information about whether the call failed: reading a source file
# that contains `except Exception as e:` is a success, and a grep that returns
# `AssertionError` did its job. These tools are flagged only when the harness
# says the call failed.
READ_ONLY_TOOL_PATTERNS = [
    re.compile(r"(?:^|_)(?:read|glob|grep|ls|list|search|recall|snapshot|overview|status|profile|neighbors|relationships|path)(?:$|_)"),
]

# Tools whose read-only-ness is worth pinning explicitly, so the suffix rule
# above cannot silently claim a writer by accident.
READ_ONLY_TOOL_NAMES = {"LS", "NotebookRead"}


def is_read_only_tool(name: str) -> bool:
    """True for tools whose result is content, and whose result text therefore
    cannot evidence a failure."""
    if name in READ_ONLY_TOOL_NAMES:
        return True
    # MCP tools arrive as `mcp____<server>_<tool>`; test each meaningful token
    # so `vault_read`, `memory_read`, `backlog_get_task` all resolve.
    return any(pat.search(name.lower()) for pat in READ_ONLY_TOOL_PATTERNS)


def output_mentions_errors(tool_name: str, content_text: str) -> bool:
    """True if the result text uses failure vocabulary. Descriptive only —
    never promotes `is_error`.

    Always False for content-returning tools: for those, arbitrary text *is* the
    expected result, so the field would be true for most calls and would teach a
    reader to distrust it.
    """
    if is_read_only_tool(tool_name):
        return False
    body = MENTION_EXCLUDE_EXIT_TRAILER.sub("", content_text)
    for pat in MENTION_ERROR_PATTERNS:
        if pat.search(body):
            return True
    return False


def has_semantic_error(content_text: str) -> bool:
    """Legacy accessor for the mention sweep (see MENTION_ERROR_PATTERNS)."""
    return output_mentions_errors("", content_text)


def parse_exit_code(content_text: str) -> int | None:
    """Exit state a tool reported in its result body, or None if it reported
    none. Absence is not corroboration in either direction."""
    match = EXIT_CODE_RE.search(content_text.rstrip())
    return int(match.group(1)) if match else None


def structured_error_body(content_text: str) -> bool:
    """True if the whole result body is a JSON object reporting failure: a
    truthy `error`, or a `code` that is not a zero. Fallback corroboration for
    sessions that predate the `stats` field on tool messages; it reads
    structure, never prose.

    A zero `code` is a success and must not promote — `{"code": 0}` appears in
    tool bodies that returned data fine."""
    try:
        parsed = json.loads(content_text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    if parsed.get("error"):
        return True
    code = parsed.get("code")
    if code is None or isinstance(code, bool):
        return False
    if isinstance(code, (int, float)):
        return code != 0
    if isinstance(code, str):
        try:
            return int(code) != 0
        except ValueError:
            return bool(code.strip())
    return False


# ── Failure classification (backlog #500) ────────────────────────────────────
#
# `stats.is_error` is a dispatch marker, not a verdict about the call: it comes
# off the harness tool-result event and fires on *any* non-zero shell exit and on
# a SIGTERM-killed command. `error_source` only says which channel flagged the
# step, and in practice there is one channel — measured over
# `_pipeline/trajectories/*.jsonl` 2026-09-09→09-17, 2,330 error steps carried
# `error_source` = {protocol: 2327, exit_code: 3}, so any gate keyed on that
# field admits 2,330 of 2,330 and discriminates nothing. What decides whether a
# failure is worth mining is its *shape*, so the shape is persisted per step as
# `failure_class`.
#
# Precedence, most specific first. Counts are flagged tool messages in
# `~/lloyd/sessions/*.json` read by payload shape at the time of writing:
#
#   timeout_or_signal  exit code < 0 — a signal death; SIGTERM (-15) is the
#                      signature of a real Bash timeout (29 rows).
#   harness_block      the harness refused or lost the call and said so in its
#                      own short form: disabled-by-configuration, safety denial,
#                      cancelled by user, no server claims the tool, transport
#                      error, argument validation (39 + ~360).
#   structured_error   the body is a JSON object reporting failure (1,586).
#   nonzero_exit       the ONLY evidence is a positive `[exit code: N]`. A
#                      `grep` with no match exits 1, `ls` on a missing path
#                      exits 2, an intentionally failing `pytest` exits 1 —
#                      1,732 rows, the largest class and the one the mining
#                      chain has already ruled non-failure.
#   protocol_flagged   the harness flagged it and the payload carries no shape
#                      evidence either way.
#
# Exactly one class is dropped downstream: `nonzero_exit` (see
# `mine-trajectories.py::is_corroborated_error`). Dropping everything the harness
# flagged would repeat, in the other direction, the mistake the pre-#389 reading
# made — the signal deaths and structured bodies above are real failures.
FAILURE_CLASSES = ("timeout_or_signal", "harness_block", "structured_error",
                   "nonzero_exit", "protocol_flagged")

# Dropped by the mining gate; named here so the two sides name it once.
NON_PROMOTABLE_FAILURE_CLASS = "nonzero_exit"

# A harness refusal is one short line the harness itself wrote, never a tool's
# multi-KB output that happens to contain the words. `.+?: no server claims tool`
# is prefixed with the tool name (`name: no server claims tool 'name'`).
HARNESS_BLOCK_MAX_LEN = 400
HARNESS_BLOCK_RE = re.compile(
    r"^(?:"
    r"tool\s+'[^']+'\s+is\s+disabled\s+by\s+configuration"
    r"|tool\s+call\s+denied:"
    r"|tool\s+'[^']+'\s+cancelled\s+by\s+user"
    r"|tool\s+call\s+arguments\s+could\s+not\s+be\s+parsed\s+as\s+json"
    r"|input\s+validation\s+error:"
    r"|.+?:\s*no\s+server\s+claims\s+tool"
    r"|.+?:\s*transport\s+error:"
    r")",
    re.IGNORECASE,
)


def classify_failure(content_text: str, exit_code: int | None) -> str:
    """Class of a step the extractor already flagged as failed.

    Derived from the payload's shape and the signed exit code, never from the
    words in the output — the keyword sweep is `output_mentions_errors`, a
    separate descriptive field that promotes nothing (#389).

    Four questions, in the order they are answerable:

    1. Was the process killed? A negative exit code is a signal, and it is the
       only class here that says *how* the call ended. It is asked first because
       the two signals genuinely co-occur — a killed call's body can itself read
       as a JSON error — and "was it killed?" is the question a mitigation
       answers.
    2. Did the call ever run? A refusal or a lost dispatch has no tool behaviour
       to mine; the harness wrote the whole body, so it is short and formulaic.
       Asked before the JSON test because `name: no server claims tool 'name'`
       arrives as a JSON body too, and `harness_block` says more about it than
       `structured_error` does.
    3. Did the tool report failure as data? A JSON object with a truthy `error`,
       read with the exit trailer removed — `structured_error_body()` parses the
       *whole* body, so a Bash-shaped `{"error": …}\n\n[exit code: 3]` would
       otherwise fall through to the exit class on a formatting artifact.
    4. Did it only exit non-zero? That is `nonzero_exit`, the one class the
       mining gate rejects.

    Anything else keeps `protocol_flagged`: a flag with no shape to argue with,
    named instead of folded into a class that claims a mechanism.
    """
    if exit_code is not None and exit_code < 0:
        return "timeout_or_signal"
    # The exit trailer is tool output, not part of the payload's shape.
    body = MENTION_EXCLUDE_EXIT_TRAILER.sub("", content_text).strip()
    if len(body) <= HARNESS_BLOCK_MAX_LEN and HARNESS_BLOCK_RE.match(body):
        return "harness_block"
    if structured_error_body(body):
        return "structured_error"
    if exit_code is not None and exit_code > 0:
        return NON_PROMOTABLE_FAILURE_CLASS
    return "protocol_flagged"


def categorize_error(text: str) -> str:
    for name, pattern in ERROR_CATEGORIES:
        if pattern.search(text):
            return name
    return "logic"


# ── Data scrubbing ────────────────────────────────────────────────────────────

def mask_sensitive(value: str) -> str:
    """Replace sensitive tokens in a string with [MASKED]."""
    for pat in SENSITIVE_PATTERNS:
        value = pat.sub("[MASKED]", value)
    return value


def _command_verb(value: str) -> str:
    """The leading program token of a shell command, bounded.

    Deliberately the *first* token and nothing cleverer: `cd x && python3 y`
    keys on `cd`, which is what the miner already keyed on before the command
    was ever truncated. Picking a better token than the one the un-truncated
    path picks is a separate change (`Bash/cd_signature`, backlog #391).
    """
    parts = value.split(None, 1)
    return parts[0][:COMMAND_HEAD_KEEP] if parts else ""


def scrub_value(key: str, value) -> object:
    """Scrub a single argument value."""
    if not isinstance(value, str):
        return value

    # Mask sensitive patterns first
    value = mask_sensitive(value)

    # File content keys — truncate aggressively
    if key.lower() in CONTENT_ARG_KEYS and len(value) > MAX_FILE_CONTENT_LEN:
        return value[:MAX_FILE_CONTENT_LEN] + f" [truncated: {len(value)} chars]"

    # Large strings — truncate
    if len(value) > MAX_STRING_LEN:
        if key.lower() in COMMAND_ARG_KEYS:
            verb = _command_verb(value)
            if verb:
                return f"{verb} [truncated: {len(value)} chars]"
        return f"[truncated: {len(value)} chars]"

    return value


def scrub_params(arguments: dict) -> dict:
    """Produce a sanitized params_summary dict from tool arguments."""
    if not isinstance(arguments, dict):
        return {}
    return {k: scrub_value(k, v) for k, v in arguments.items()}


def extract_result_text(content) -> str:
    """Pull plain text from a toolResult content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def result_summary(content, is_error: bool) -> str:
    text = extract_result_text(content)
    if is_error:
        preview = text[:MAX_ERROR_LEN].replace("\n", " ").strip()
        return f"ERROR: {preview}"
    return f"OK: {len(text)} chars"


# ── Signal extraction ─────────────────────────────────────────────────────────

SIGNAL_RE = re.compile(r"SIGNAL:([A-Z_]+)")

# The only role whose text counts as the run emitting a signal. Skills instruct
# the emission in prose ("stop and SIGNAL:BLOCKED"), so what a run does when it
# blocks is write that token in one of its own messages. Every other role is
# text the run was handed rather than authored: in an autonomy run the first
# `user` message IS the dispatched SKILL.md body, so a skill that documents
# `SIGNAL:BLOCKED` in its error-handling section handed that token to every one
# of its runs forever, whatever happened — measured on 2026-09-17, 9 of 9
# `BLOCKED` rows were inherited this way and none was emitted (#1238). A `tool`
# result is a subprocess's stdout, which the run merely carried: that is a real
# emission channel (`scripts/memory/process-groundskeeper-queue.py` prints
# `SIGNAL:TASK_COMPLETE`) and it is dropped here by choice, with a person
# deciding whether to keep it — which is why the excluded hits are returned
# rather than discarded.
EMITTED_SIGNAL_ROLES = frozenset({"assistant"})


def collect_signals(messages: list[dict]) -> tuple[list[str], dict[str, list[str]]]:
    """Split `SIGNAL:<X>` hits by the role whose text carried them.

    Returns `(signals, inherited)`, both in first-seen order and de-duplicated:

    * `signals` — tokens found in a role listed in `EMITTED_SIGNAL_ROLES`. The
      only field a consumer may read as an outcome of this run.
    * `inherited` — `{role: [tokens]}` for the hits that rule excluded, so a run
      that inherited three `BLOCKED`s from its own skill body reads as
      `signals: []` plus `inherited_signals: {"user": ["BLOCKED"]}`, never as
      silently empty. A token the run *did* emit is not also reported as
      inherited, even when a skill body or a tool result said it too.

    Only a message's `content` is read. A `SIGNAL:` string inside an assistant
    `tool_calls` argument (a shell command that echoes the token) is not an
    emission: the emission is what the command printed, which lands in the tool
    result, and counting the command would hand the field back to any skill that
    documents such a command.

    `SIGNAL_RE`'s capture group stops at the next colon, so the checkpoint form
    in `skills/.archived/python-library-dev/SKILL.md` — `SIGNAL:CHECKPOINT:
    PLAN_COMPLETE` — is collected as `CHECKPOINT` and loses the specific
    checkpoint. That is pre-existing and deliberate here: #1238 changed which
    roles are read, not what one hit captures. A test pins it.
    """
    signals: list[str] = []
    emitted: set[str] = set()
    inherited: dict[str, list[str]] = {}
    inherited_seen: dict[str, set[str]] = {}

    for msg in messages:
        role = str(msg.get("role") or "unknown")
        text = extract_result_text(msg.get("content", ""))
        for match in SIGNAL_RE.finditer(text):
            sig = match.group(1)
            if role in EMITTED_SIGNAL_ROLES:
                if sig not in emitted:
                    emitted.add(sig)
                    signals.append(sig)
                continue
            seen = inherited_seen.setdefault(role, set())
            if sig not in seen:
                seen.add(sig)
                inherited.setdefault(role, []).append(sig)

    for role in [r for r, sigs in inherited.items() if all(s in emitted for s in sigs)]:
        del inherited[role]
    return signals, inherited


# ── Session classification ───────────────────────────────────────────────────

def classify_session(data: dict, session_id: str | None = None) -> str:
    """Class of the session from the session JSON's own fields.

    Takes the parsed object, not a path: the filename is exactly what must not be
    an input (#493 clauses 1-2). `session_id` is a fallback for a caller that has
    a key but not the file's own field — the miner's join to
    `~/lloyd-data/sessions/<session_key>.json`, whose rows are keyed on the name and
    whose legacy rows predate any id being emitted.

    Three decisions, in the order they are applied (#1143):

    1. `app.sessions_io.is_user_session` decides admission, and it decides it
       alone: a platform on its deny-list is never a human class here, whatever
       `SESSION_CLASS` says about it. The extractor keeps no second list of which
       platforms are machines, because the two would drift the first time a
       platform was added to one and not the other — and the drift direction that
       matters is a machine platform reading as Alan's chat.
    2. A platform the backend records is then decided by `SESSION_CLASS`, so the
       `e2e-harness` smokes stay `smoke` whatever their id looks like. A platform
       nobody has heard of is `unknown`: reported, and not admitted.
    3. A session that survives as a human class whose id is not the three-part
       chat shape is `test`: `v2-*`, `dbg*`, `soak*`, `*_selftest_*` and four-part
       sandbox ids are all created through the chat endpoint with
       `platform: mission-control` and were, until now, the only thing the mining
       corpus contained.

    `inner_voice` decides nothing. It is the switch that enables the observer for
    a human's chat, so reading it as "the observer took this turn" is what made
    the corpus exclude the human chats (#493's wrong premise, closed by #1143).
    """
    platform = data.get("platform")
    cls = SESSION_CLASS.get(platform) or UNKNOWN_CLASS
    if not is_user_session(data):
        # A machine platform. `worker`/`autonomy` by name, anything else
        # `unknown` — and never a class the miner admits, even if SESSION_CLASS
        # mapped it to one: admission is is_user_session's to decide.
        return UNKNOWN_CLASS if cls in HUMAN_CLASSES else cls
    if cls in HUMAN_CLASSES:
        sid = str(session_id or data.get("session_id")
                  or data.get("id") or "")
        if sid and not CHAT_SESSION_ID_RE.match(sid):
            return TEST_CLASS
    return cls


# ── Session parsing ──────────────────────────────────────────────────────────

def parse_session(path: Path) -> dict | None:
    """
    Parse a single Hermes session JSON file.

    Hermes format: single JSON object with top-level messages array.
    Messages use role "user"/"assistant"/"tool" with tool_calls on assistant
    messages and tool_call_id on tool messages.

    Returns a trajectory dict or None if the session has no tool calls.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    session_id = data.get("session_id", path.stem)
    session_ts = data.get("session_start", "") or data.get("created_at", "")
    messages = data.get("messages", [])

    # `agent_id` is the legacy field: derived from the filename, and wrong for
    # every session in the live store (#493). It survives only for historical
    # corpus rows and for anyone still filtering on it; `session_class` below is
    # what decides whether a session is human-initiated work.
    agent_id = "autonomy" if path.stem.startswith("autonomy_") else "lloyd"
    session_class = classify_session(data, session_id=session_id)

    # Build call_id → tool_call map from assistant messages
    call_map: dict[str, dict] = {}
    call_sequence: dict[str, int] = {}
    sequence_counter = 0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            call_id = tc.get("id") or tc.get("call_id", "")
            if not call_id:
                continue
            # Normalize to shared structure
            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {"_raw": raw_args[:MAX_STRING_LEN]}
            call_map[call_id] = {
                "name": func.get("name", "unknown"),
                "arguments": raw_args,
            }
            call_sequence[call_id] = sequence_counter
            sequence_counter += 1

    # Build call_id → result map from tool messages
    result_map: dict[str, dict] = {}
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id", "")
        if call_id:
            content = msg.get("content", "")
            # Normalize content to string (lloyd sessions use [{type,text}] blocks)
            content_text = extract_result_text(content)
            # `stats.is_error` is set by the harness at dispatch time and is
            # present on every tool message in the current session format, so it
            # is the flag we read — but it is a *dispatch* marker, not a verdict
            # about the call: it is true for any non-zero shell exit and for a
            # SIGTERM-killed command, classes the mining chain has ruled
            # non-failure. It says "the harness flagged this", never "this is
            # worth mining"; `classify_failure()` below decides that from the
            # payload shape. It replaced the old body-regex derivation (#392);
            # it did not replace the need for a class.
            stats = msg.get("stats")
            stats_error = stats.get("is_error") if isinstance(stats, dict) else None
            result_map[call_id] = {
                "content": content_text,
                "stats_error": stats_error,
                "structured_error": structured_error_body(content_text),
            }

    if not call_map:
        return None

    # Build unified tool list sorted by sequence
    tools: list[dict] = []
    error_tools: list[dict] = []

    for call_id, call in sorted(call_map.items(), key=lambda x: call_sequence[x[0]]):
        name = call["name"]
        raw_args = call["arguments"]
        params = scrub_params(raw_args) if isinstance(raw_args, dict) else {}

        result = result_map.get(call_id)
        if result is not None:
            content_text = result.get("content") or ""
            exit_code = parse_exit_code(content_text)
            mentions = output_mentions_errors(name, content_text)
            stats_error = result.get("stats_error")
            if stats_error is None:
                # Session predates the `stats` field: fall back to the
                # structured-body reading, which is still structure, not prose.
                stats_error = bool(result.get("structured_error"))
            protocol_error = bool(stats_error)
            exit_error = exit_code is not None and exit_code != 0
            # Corroboration only. Whether the output *reads* like a failure is
            # `output_mentions_errors`, recorded but never promoted (#389).
            is_error = protocol_error or exit_error
            res_summary = result_summary(content_text, is_error)
            if protocol_error:
                error_source = "protocol"
            elif exit_error:
                error_source = "exit_code"
            else:
                error_source = None
            # `error_source` records the channel that flagged the step and is
            # almost always "protocol"; `failure_class` records what the failure
            # *is*, which is the only thing a downstream gate can grade (#500).
            failure_class = (classify_failure(content_text, exit_code)
                             if is_error else None)
        else:
            is_error = False
            res_summary = "OK: no result recorded"
            error_source = None
            failure_class = None
            exit_code = None
            mentions = False
            content_text = ""

        seq = call_sequence[call_id]
        entry = {
            "name": name,
            "params_summary": params,
            "result_summary": res_summary,
            "is_error": is_error,
            "error_source": error_source,
            "failure_class": failure_class,
            "output_mentions_errors": mentions,
            "exit_code": exit_code,
            "call_id": call_id,
            "sequence": seq,
        }
        tools.append(entry)

        if is_error:
            error_tools.append({
                "name": name,
                "sequence": seq,
                "error_type": categorize_error(content_text),
                "error_source": error_source,
                "failure_class": failure_class,
                "exit_code": exit_code,
                "params_summary": params,
            })

    error_count = len(error_tools)

    # Use file mtime as fallback timestamp
    if not session_ts:
        mtime = os.path.getmtime(path)
        session_ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    # Signals, split by who said them: `signals` is the run's own emission,
    # `inherited_signals` the vocabulary it was handed (the dispatched skill body
    # in the first `user` message, tool results). See `collect_signals`.
    signals, inherited_signals = collect_signals(messages)

    return {
        "session_key": session_id,
        "agent_id": agent_id,
        "session_class": session_class,
        # Which loop ran it, when the backend recorded one: `autotriage`,
        # `autocode`, `autonomy-task:68`… What makes a dropped session
        # attributable instead of an anonymous count.
        "session_source": data.get("source"),
        "timestamp": session_ts,
        "tool_count": len(tools),
        "error_count": error_count,
        "has_errors": error_count > 0,
        "tools": tools,
        "error_tools": error_tools,
        "signals": signals,
        # `{role: [tokens]}` for the hits `signals` refused, so an inherited
        # token is recoverable from the row instead of lost. Named per role
        # because the role is the whole finding: `user` here means the skill body
        # that was dispatched to this run, `tool` a subprocess's stdout.
        "inherited_signals": inherited_signals,
    }


# ── Watermark state ───────────────────────────────────────────────────────────

def load_watermark() -> dict:
    if WATERMARK_PATH.exists():
        try:
            return json.loads(WATERMARK_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "last_run": None,
        "sessions_processed": 0,
        "last_session_mtime": None,
    }


def save_watermark(state: dict) -> None:
    WATERMARK_PATH.write_text(json.dumps(state, indent=2))


# ── Session discovery ─────────────────────────────────────────────────────────

def discover_sessions(agent_filter: str | None = None) -> list[Path]:
    """Return sorted list of session JSON paths from <data root>/sessions/.

    agent_filter:
      'worker' / 'autonomy' → only autonomy_*.json files
      'main'   / 'lloyd'    → only non-autonomy *.json files
      None                  → all sessions
    """
    paths: list[Path] = []
    if LLOYD_SESSIONS.exists():
        for p in LLOYD_SESSIONS.glob("*.json"):
            is_autonomy = p.stem.startswith("autonomy_")
            if agent_filter in ("worker", "autonomy"):
                if is_autonomy:
                    paths.append(p)
            elif agent_filter in ("main", "lloyd"):
                if not is_autonomy:
                    paths.append(p)
            else:
                paths.append(p)
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def filter_by_mtime(paths: list[Path], since_mtime: float | None) -> list[Path]:
    if since_mtime is None:
        return paths
    return [p for p in paths if p.stat().st_mtime > since_mtime]


def filter_by_days(paths: list[Path], days: int) -> list[Path]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    cutoff_ts = cutoff.timestamp()
    return [p for p in paths if p.stat().st_mtime >= cutoff_ts]


# ── Output writing ────────────────────────────────────────────────────────────

def trajectory_date_key(traj: dict) -> str:
    """Return YYYY-MM-DD in LOCAL_TZ (America/Los_Angeles) from trajectory
    timestamp, defaulting to today. Buckets must match daily-note dates
    (local), not UTC — sessions after 17:00 PDT otherwise land one day late."""
    ts = traj.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")


def append_trajectories(trajectories: list[dict]) -> None:
    """Append trajectories to date-bucketed output files.

    Idempotent per session_key: entries whose session_key already exists
    in the target file are skipped (dedup-on-write). Backfill re-covers
    sessions a prior run already wrote (watermark/late-end timing) and a
    plain append produced byte-identical duplicate pairs (08-28/08-29
    defect, 6th cycle). session_key is unique per session, so dedup is
    safe.
    """
    # Group by date
    by_date: dict[str, list[dict]] = {}
    for traj in trajectories:
        date_key = trajectory_date_key(traj)
        by_date.setdefault(date_key, []).append(traj)

    for date_key, items in by_date.items():
        out_path = OUTPUT_DIR / f"{date_key}.jsonl"
        existing_keys: set[str] = set()
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_keys.add(json.loads(line).get("session_key", ""))
                    except (ValueError, KeyError):
                        pass
        with open(out_path, "a", encoding="utf-8") as fh:
            for item in items:
                key = item.get("session_key", "")
                if key and key in existing_keys:
                    continue
                if key:
                    existing_keys.add(key)
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def rewrite_trajectories(trajectories: list[dict]) -> None:
    """Write trajectories to date-bucketed output files (overwrite mode)."""
    # Group by date
    by_date: dict[str, list[dict]] = {}
    for traj in trajectories:
        date_key = trajectory_date_key(traj)
        by_date.setdefault(date_key, []).append(traj)

    # Remove existing output files that will be rewritten
    existing = list(OUTPUT_DIR.glob("*.jsonl"))
    for p in existing:
        if p.name != ".watermark.json":
            p.unlink(missing_ok=True)

    for date_key, items in by_date.items():
        out_path = OUTPUT_DIR / f"{date_key}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats() -> None:
    """Print summary statistics from existing trajectory files."""
    files = sorted(OUTPUT_DIR.glob("*.jsonl"))
    if not files:
        print("No trajectory files found.")
        return

    total_sessions = 0
    total_tools = 0
    total_errors = 0
    agent_counts: dict[str, int] = {}
    session_class_counts: dict[str, int] = {}
    error_type_counts: dict[str, int] = {}
    error_source_counts: dict[str, int] = {}
    failure_class_counts: dict[str, int] = {}
    tool_name_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    traj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                total_sessions += 1
                total_tools += traj.get("tool_count", 0)
                total_errors += traj.get("error_count", 0)

                agent = traj.get("agent_id", "unknown")
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
                # `uncoded` is what a pre-#493 row reads as: no emitted class. It
                # is reported, never folded into `interactive`.
                sclass = traj.get("session_class") or "uncoded"
                session_class_counts[sclass] = session_class_counts.get(sclass, 0) + 1

                for et in traj.get("error_tools", []):
                    etype = et.get("error_type", "unknown")
                    error_type_counts[etype] = error_type_counts.get(etype, 0) + 1
                    source = et.get("error_source")
                    if source:
                        error_source_counts[source] = error_source_counts.get(source, 0) + 1
                    fclass = et.get("failure_class")
                    if fclass:
                        failure_class_counts[fclass] = failure_class_counts.get(fclass, 0) + 1

                for tool in traj.get("tools", []):
                    tname = tool.get("name", "unknown")
                    tool_name_counts[tname] = tool_name_counts.get(tname, 0) + 1
                    source = tool.get("error_source")
                    if source:
                        error_source_counts[source] = error_source_counts.get(source, 0) + 1

                # Emitted tokens only, because `signals` now holds only those.
                # The histogram reported 9 `BLOCKED` for 2026-09-17, a day with no
                # genuine emission at all, because it was counting what the skill
                # body said (#1238). Rows written before that change still carry
                # inherited tokens in `signals` and are counted as written: the
                # role is not recoverable from a row that never recorded it, so a
                # historical bucket is not directly comparable to a fresh one.
                for sig in traj.get("signals", []):
                    signal_counts[sig] = signal_counts.get(sig, 0) + 1

    watermark = load_watermark()

    print("=" * 60)
    print("TRAJECTORY STATS")
    print("=" * 60)
    print(f"  Output files:       {len(files)}")
    print(f"  Total sessions:     {total_sessions}")
    print(f"  Total tool calls:   {total_tools}")
    print(f"  Total errors:       {total_errors}")
    if total_tools > 0:
        print(f"  Error rate:         {total_errors / total_tools:.1%}")
    print()
    print("By agent:")
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        print(f"  {agent:<20} {count}")
    # Printed alongside the agent tally, not instead of it. The agent histogram is
    # the one that has read as a single value on this machine for the corpus's whole
    # life (`agent_id` comes from the filename); the class histogram is the one that
    # can show the loop's share of the corpus, which is what #493 is about.
    print()
    print("By session class:")
    for sclass, count in sorted(session_class_counts.items(), key=lambda x: -x[1]):
        print(f"  {sclass:<20} {count}")
    print()
    print("Top tools:")
    for name, count in sorted(tool_name_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name:<30} {count}")
    print()
    print("Error types:")
    for etype, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<20} {count}")
    print()
    print("Error sources:")
    for src, count in sorted(error_source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<20} {count}")
    # Counted from `error_tools` only, so it is one row per failed step — the
    # source tally above counts every failed step twice, once from each list it
    # appears in. `protocol` names the channel that flagged a step and is nearly
    # always the same value; this histogram is the one that says what the failure
    # was, which is what the post-landing over-count is measured on (#500).
    print()
    print("Failure classes:")
    for cls, count in sorted(failure_class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<20} {count}")
    print()
    print("Signals seen:")
    for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
        print(f"  {sig:<30} {count}")
    print()
    print("Watermark:")
    print(f"  last_run:            {watermark.get('last_run', 'never')}")
    print(f"  sessions_processed:  {watermark.get('sessions_processed', 0)}")
    print(f"  last_session_mtime:  {watermark.get('last_session_mtime', 'none')}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract trajectory logs from Lloyd session JSON files."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Reprocess all sessions (ignore watermark)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Process only sessions modified in the last N days"
    )
    parser.add_argument(
        "--agent", type=str, default=None,
        help="Filter to a specific agent (e.g. worker, main)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print summary statistics from existing trajectory files"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stats:
        print_stats()
        return

    watermark = load_watermark()

    # Discover sessions
    all_sessions = discover_sessions(agent_filter=args.agent)

    # Determine filter
    if args.full:
        sessions_to_process = all_sessions
    elif args.days is not None:
        sessions_to_process = filter_by_days(all_sessions, args.days)
    else:
        # Incremental: only sessions newer than watermark
        last_mtime_str = watermark.get("last_session_mtime")
        if last_mtime_str:
            try:
                last_mtime_dt = datetime.fromisoformat(last_mtime_str.replace("Z", "+00:00"))
                last_mtime_ts = last_mtime_dt.timestamp()
                sessions_to_process = filter_by_mtime(all_sessions, last_mtime_ts)
            except (ValueError, AttributeError):
                sessions_to_process = all_sessions
        else:
            sessions_to_process = all_sessions

    print(f"Processing {len(sessions_to_process)} session(s)...", file=sys.stderr)

    trajectories: list[dict] = []
    max_mtime: float = 0.0
    skipped = 0
    failed = 0

    for path in sessions_to_process:
        try:
            traj = parse_session(path)
            if traj is None:
                skipped += 1
                continue
            trajectories.append(traj)
            mtime = path.stat().st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
        except Exception as exc:
            print(f"  ERROR parsing {path}: {exc}", file=sys.stderr)
            failed += 1

    print(f"  Parsed: {len(trajectories)}  Skipped (no tools): {skipped}  Failed: {failed}", file=sys.stderr)

    # Write output
    if trajectories:
        if args.full:
            rewrite_trajectories(trajectories)
        else:
            append_trajectories(trajectories)

    # Update watermark
    now = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    new_mtime_str = None
    if max_mtime > 0:
        new_mtime_str = datetime.fromtimestamp(max_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    updated_watermark = {
        "last_run": now,
        "sessions_processed": watermark.get("sessions_processed", 0) + len(trajectories),
        "last_session_mtime": new_mtime_str or watermark.get("last_session_mtime"),
    }
    save_watermark(updated_watermark)

    print(f"  Written to: {OUTPUT_DIR}", file=sys.stderr)
    print(f"  Watermark updated: {now}", file=sys.stderr)


if __name__ == "__main__":
    main()
