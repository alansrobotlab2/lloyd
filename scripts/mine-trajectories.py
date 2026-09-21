#!/usr/bin/env python3
"""
mine-trajectories.py — Mine trajectory JSONL files for skill candidates.

Reads trajectory files from Phase 1 (extract-trajectories.py) and produces
markdown candidate files for skill patterns (both error and success patterns).

Output: ~/obsidian/skills/candidates/candidate-{pattern-slug}-{YYYYMMDD}.md
"""

import argparse
import json
import os
import re
import sys
from bisect import insort
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ── Paths ────────────────────────────────────────────────────────────────────

TRAJECTORY_DIR = Path.home() / "lloyd" / "_pipeline" / "trajectories"
OUTPUT_DIR = Path.home() / "lloyd" / "_pipeline" / "skills" / "candidates"


def set_trajectory_dir(path: Path) -> None:
    """Override trajectory directory for testing."""
    global TRAJECTORY_DIR
    TRAJECTORY_DIR = path


# ── Error categorization ─────────────────────────────────────────────────────

ERROR_CATEGORIES = [
    ("permission", re.compile(r"permission denied|access denied|forbidden|EPERM|EACCES", re.IGNORECASE)),
    ("not_found", re.compile(r"file not found|no such file|not found|404|ENOENT", re.IGNORECASE)),
    ("timeout", re.compile(r"timeout|timed out|ETIMEDOUT|deadline exceeded", re.IGNORECASE)),
    ("network", re.compile(r"connection refused|ECONNREFUSED|DNS|ENOTFOUND|network|EHOSTUNREACH", re.IGNORECASE)),
    ("validation", re.compile(r"invalid|malformed|parse error|syntax error|schema|validation", re.IGNORECASE)),
    ("resource", re.compile(r"out of memory|disk full|quota|ENOMEM|ENOSPC|resource exhausted", re.IGNORECASE)),
]


def categorize_error(text: str) -> str:
    """Categorize an error message into a type."""
    if not text:
        return "logic"
    text_lower = text.lower()
    for name, pattern in ERROR_CATEGORIES:
        if pattern.search(text):
            return name
    return "logic"


def categorize_result_summary(result_summary: str) -> str:
    """Categorize based on result_summary field."""
    if not result_summary:
        return "logic"
    result_lower = result_summary.lower()
    for name, pattern in ERROR_CATEGORIES:
        if pattern.search(result_lower):
            return name
    return "logic"


# ── Tool name normalization (for sequence mining) ────────────────────────────

BASH_CMD_CATEGORIES = {
    "explore": {"ls", "find", "cat", "head", "tail", "wc", "file", "stat", "du", "tree"},
    "modify": {"sed", "awk"},
    "git": {"git"},
    "python": {"python3", "python", "pip", "uv"},
    "http": {"curl", "wget"},
    "container": {"docker", "podman"},
    "service": {"supervisorctl", "systemctl", "journalctl"},
    "fs": {"cd", "mkdir", "rm", "mv", "cp", "chmod", "chown", "ln", "touch"},
}

MCP_PREFIX_MAP = {
    "mcp____vault_": "mcp:vault",
    "mcp____fact_": "mcp:fact",
    "mcp____backlog_": "mcp:backlog",
    "mcp____autonomy_": "mcp:autonomy",
    "mcp____http_": "mcp:http",
    "mcp____browser_": "mcp:browser",
    "mcp____email_": "mcp:email",
    "mcp____skills_": "mcp:skills",
    "mcp____pipeline_": "mcp:pipeline",
    "mcp____memory_": "mcp:memory",
    "mcp____chat_": "mcp:chat",
    "mcp____calendar_": "mcp:calendar",
    "mcp____discord_": "mcp:discord",
}

BUILTIN_TOOLS = {
    "Read": "read", "Edit": "edit", "Write": "write",
    "Glob": "glob", "Grep": "grep", "Agent": "agent",
    "ToolSearch": "toolsearch", "Skill": "skill",
    "WebFetch": "webfetch", "WebSearch": "websearch",
    "TodoWrite": "todowrite", "NotebookEdit": "notebook",
}


def normalize_tool_name(tool: dict) -> str:
    """
    Produce a semantic label for a tool call.
    Bash commands are sub-categorized by command type.
    MCP tools are grouped by server prefix.
    Error state is appended as :ERR.
    """
    name = tool.get("name", "unknown")
    is_error = tool.get("is_error", False)

    # Built-in tools
    if name in BUILTIN_TOOLS:
        label = BUILTIN_TOOLS[name]
    elif name == "Bash":
        cmd = tool.get("params_summary", {}).get("command", "")
        first_word = cmd.split()[0] if cmd and cmd.split() else ""
        # Strip path prefix (e.g., /usr/bin/python3 -> python3)
        first_word = first_word.rsplit("/", 1)[-1]
        label = "bash:other"
        for category, commands in BASH_CMD_CATEGORIES.items():
            if first_word in commands:
                label = f"bash:{category}"
                break
    else:
        # MCP tools
        label = None
        for prefix, mapped in MCP_PREFIX_MAP.items():
            if name.startswith(prefix):
                label = mapped
                break
        if label is None:
            if name.startswith("mcp____"):
                label = "mcp:other"
            else:
                label = name.lower()

    if is_error:
        label += ":ERR"
    return label


# ── Pattern matching ─────────────────────────────────────────────────────────

# A scrubbed value that says "there was something here" and nothing else:
# `[truncated: 900 chars]`, `[MASKED]`, or an extraction artefact of the same
# shape. The prefix is what both writers emit, so a leading `[` is the test.
KEY_ARTIFACT_PREFIX = "["


def _command_key(cmd: str) -> str:
    """Grouping key for a shell call: the program it ran.

    Rows written before backlog #391 hold a bare `[truncated: N chars]` where
    the command was, and that placeholder was being taken as the program name —
    which is how 625 unrelated calls (transcript extraction, awk, nvidia-smi,
    git) became the single #2-ranked pattern in the corpus. Such a call has no
    recoverable key, so it returns "" and the caller drops it rather than
    filing it into one shared bucket.
    """
    parts = cmd.split()
    if not parts:
        return ""
    head = parts[0].rsplit("/", 1)[-1]
    if not head or head.startswith(KEY_ARTIFACT_PREFIX):
        return ""
    return f"cmd:{head}"


def normalize_params_signature(params_summary: dict) -> str:
    """
    Create a normalized signature from params_summary for grouping.
    Uses tool name + key parameter patterns.

    "" means the call cannot be keyed and must be dropped from signature mining,
    not filed under a shared fallback bucket.
    """
    if not params_summary:
        return ""

    # For run_bash, extract command pattern
    if "command" in params_summary:
        cmd = params_summary["command"]
        if isinstance(cmd, str):
            # Normalize paths and specific values
            cmd = re.sub(r'/home/[^\s]+', '/home/USER', cmd)
            cmd = re.sub(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', 'DATE', cmd)
            cmd = re.sub(r'[0-9]{2}:[0-9]{2}:[0-9]{2}', 'TIME', cmd)
            return _command_key(cmd)
        return "run_bash_signature"

    # For file operations, extract operation + pattern
    if "path" in params_summary:
        path = str(params_summary["path"])
        # Extract file extension pattern
        ext_match = re.search(r'\.([a-zA-Z0-9]+)$', path)
        if ext_match:
            return f"file_{ext_match.group(1)}_signature"
        return "file_signature"

    if "pattern" in params_summary:
        return f"pattern_{str(params_summary['pattern'])[:20]}_signature"

    # Generic signature based on keys
    keys = sorted(params_summary.keys())
    return f"{'_'.join(keys)}_signature"


# ── Candidate emission ───────────────────────────────────────────────────────

def is_emittable(pattern: dict) -> bool:
    """Whether a mined pattern is worth a candidate file.

    A `*_signature` key names the tool plus the *set of argument names* the call
    carried — `Read/file_path_limit_offset_signature` means "Read was called
    with file_path, limit and offset" — and the body is a frequency count of
    successful calls. There is no failure, no mitigation and no decision in it,
    which authoring rule 5 forbids as skill content; 11 skills were archived
    2026-09-04 for exactly that class and one of them was re-mined within the
    week. On 2026-09-08 the class was 93 of 103 candidate keys and all 20 of the
    top 20 by occurrences, so the gate was selecting for it (#391).

    A bare program name (`cmd:cd`, `cmd:ls`) is the same non-skill wearing a
    better key, so it goes too: `Bash/cd_signature` was the #1 pattern that day
    at 1,129 occurrences. What survives is a key that names a parameter *value*
    shape carrying a decision — `read_limit_on_files_over_2000_lines` — which
    nothing currently produces, so success candidates are suppressed wholesale
    until a producer for that class exists.

    An `error` pattern is untouched: it keys on (tool, error_type) and is built
    without a `has_error_recovery` key at all, so the sequence rule below cannot
    reach it.

    A `sequence` pattern survives only when it is flagged
    `has_error_recovery: true` — some observed instance of the n-gram re-attempted
    the failed call or touched its target again, which is what `ngram_shows_recovery`
    decides. An `:ERR` step followed by any old next step is not it: since #1327
    that adjacency sets nothing. That flag is what makes a repeated n-gram a lesson
    rather than a call order: `seq-2-calendar-events-email-recent` appears in 157
    sessions with 6 of 6 steps succeeding and its top worker class at 100 % of
    them, and `seq-2-write-read` in 209 sessions at 94 % — one pipeline's
    ordinary order, which the `sessions >= 3` gate counts as a shared lesson.
    Measured 2026-09-16, 672 of the 780 actionable candidate keys were
    `has_error_recovery: false` and 108 were flagged true — by the adjacency rule
    then in force, which is why #1327 narrowed the flag: four of five keys
    adjudicated from that pool on 2026-09-21 had successors that were unrelated
    next commands. At the runbook's 5 patterns a night, adjudicating the pool by
    hand was ~156 nights to reach "no skill here" for every one of them. A
    sequence that reaches here with no flag at all stays emittable, so a
    hand-built or legacy pattern is not suppressed by absence.
    """
    if pattern.get("type") != "success":
        # `is False`, not falsy: an `error` pattern dict carries no
        # `has_error_recovery` key, and a falsy test would suppress all 116 of
        # them along with the sequences (#1181).
        if (pattern.get("type") == "sequence"
                and pattern.get("has_error_recovery") is False):
            return False
        return True
    sig = (pattern.get("params_signature") or "").strip()
    if not sig or sig == "generic":
        return False
    if sig.startswith("cmd:") or sig.endswith("_signature"):
        return False
    return True


def emit_candidates(patterns: list[dict], output_dir: Path) -> list[Path]:
    """Write every emittable pattern; return one entry per file written.

    The filter lives here and in `write_candidate_file`, not in the three
    miners: the pattern tables are still complete telemetry (`--stats` and a
    re-ranked corpus need to see what the extractor is producing), but nothing
    in the non-skill class reaches skill authoring.

    The returned list is one entry per *file*, because `main()` prints one
    `Written:` line per entry and hands the same list to `write_index()`. It gets
    there by refusing a collision before writing anything, not by folding one away:
    two patterns landing on one path raise below instead of being silently
    de-duplicated, because a folded list is how #1131's aliased sequence and #515's
    collapsed error keys both hid — a run reporting 1234 candidates over 1169 files,
    and, for the error class, 73 writes landing on 25 paths with the surviving file
    holding one arbitrary bucket's evidence. That is also why `main()` measures its
    suppression count off `is_emittable` rather than off `len(...) - len(written)`:
    the subtraction turns a refused collision into a phantom suppression (#1131,
    clause 5).

    Error patterns reach the writer already merged onto their candidate key
    (`merge_error_patterns`), which is what makes the error class 1:1 by
    construction rather than by luck: `mine_error_patterns` groups on the
    signature too and `candidate_pattern_key` deliberately drops it, so a
    per-pattern writer would put far more writes than keys at the paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    units = merge_error_patterns(patterns)
    emittable = [p for p in units if is_emittable(p)]
    # Checked over the WHOLE list, every class, BEFORE the first write. The first
    # version of this guard looked only at `type == "error"` paths and stayed silent
    # while a `candidate-seq-3-…` name was written twice by two distinct n-grams,
    # which is the review's own finding against round SM_20260914_140724 (#515 clause
    # 1: "no path appears more than once in the list returned by emit_candidates()" is
    # a statement about the list, not about one class of it). A duplicate here is
    # always the same defect — some key reaching the filename step where two patterns
    # are still equal. Checking before the loop rather than after it is what turns
    # detection into prevention: post-loop, the second write has already replaced the
    # first file's bytes and the raise only reports the damage, while the item's own
    # requirement is that a collision be "an error rather than a lost file".
    names = [candidate_filename(p) for p in emittable]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise AssertionError(
            f"{len(emittable)} emittable patterns resolve to {len(set(names))} "
            f"filenames; {len(emittable) - len(set(names))} of them would be "
            "discarded by last-writer-wins, so nothing was written. Colliding names: "
            + ", ".join(dupes[:5]))
    for pattern in emittable:
        path = write_candidate_file(pattern, output_dir)
        if path:
            written.append(Path(path))
    assert len(written) == len(emittable), (
        f"{len(emittable)} patterns passed `is_emittable` but "
        f"{len(written)} files came back: the writer refused a pattern the emitter "
        "had already accepted, so `main()` would report a file that does not exist")
    # One sequence pattern = one n-gram = one key, and `mine_sequence_patterns`
    # returns each n-gram once, so comparing distinct keys to sequence files is the
    # key-level half of the same claim: a 50-character slug that folds two n-grams
    # onto one *key* (not just one name) leaves this pair equal while the path
    # guard above can only see the filename (#1131 clause 2).
    sequence_keys = {
        candidate_pattern_key(p) for p in units
        if p.get("type") == "sequence" and is_emittable(p)
    }
    sequence_paths = [p for p in written if p.name.startswith("candidate-seq-")]
    assert len(sequence_paths) == len(sequence_keys), (
        f"{len(sequence_keys)} distinct sequence pattern keys wrote "
        f"{len(sequence_paths)} candidate-seq-*.md files: a sequence candidate "
        "path is still being written twice (#1131)"
    )
    return written


def slugify(text: str) -> str:
    """Convert text to URL-safe slug (no slashes for filenames)."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)  # Replace everything non-alphanumeric with -
    text = re.sub(r'-+', '-', text)
    text = text.strip('-')
    return text[:SLUG_CAP]  # Limit length


# The 50-character cut above is lossy, and it sat on the path of both the sequence
# *key* and the sequence *filename*: `candidate_pattern_key()` slugified the n-gram
# before building `seq-{n}-{slug}`, and `write_candidate_file()` slugified that key
# again for the name. Two distinct n-grams whose first 50 slug characters agree —
# `… -> backlog_write_task` vs `… -> backlog_write_task:ERR`, or two 5-grams
# differing only in the final tool — therefore shared one filename and one
# `pattern:` field, the second write silently replaced the first, and which n-gram
# survived was dict iteration order (measured 2026-09-15: 1674 mined sequence
# patterns -> 1671 files). #1131.
SLUG_CAP = 50


def slug_disambiguator(pre_slug_text: str) -> str:
    """8 hex chars of sha256 over the exact text `slugify` truncated.

    The import is function-local on purpose: a module-level one would shift the
    pre-existing `normalize_tool_name` redefinition finding at line 313, whose
    message quotes line 106, and the gate's normaliser strips only the
    `path:line:col:` prefix — so a moved *message* number reads as a new finding
    and fails the static rung on a change that introduced nothing.
    """
    import hashlib

    return hashlib.sha256(pre_slug_text.encode("utf-8")).hexdigest()[:8]


def slug_for(text: str) -> str:
    """`slugify(text)`, made injective on `text` when the cap actually bit.

    The suffix is added *only* when the slug reached `SLUG_CAP`. That scoping is
    the whole point: the key derived from this is also the verdict-ledger join,
    and every `seq-*` row already in `_pipeline/skills/reviews/verdicts.jsonl` is
    under the cap (53 rows / 28 distinct keys on 2026-09-15, longest 47 characters),
    so nothing existing changes meaning and no verdict is orphaned — below the cap
    this is exactly `slugify`. An unscoped widening is the #515 failure: widening a
    key with no migration left 21 coarse rows unreachable.
    """
    slug = slugify(text)
    if len(slug) >= SLUG_CAP:
        return f"{slug}-{slug_disambiguator(text)}"
    return slug


# ── Tool name normalization (for sequence mining) ────────────────────────────

_BASH_CMD_CATEGORIES = {
    "bash:explore": {"ls", "find", "cat", "head", "tail", "wc"},
    "bash:modify": {"sed", "awk"},
    "bash:git": {"git"},
    "bash:python": {"python3", "python", "pip", "uv"},
    "bash:http": {"curl", "wget"},
    "bash:container": {"docker", "podman"},
    "bash:service": {"supervisorctl", "systemctl"},
    "bash:fs": {"cd", "mkdir", "rm", "mv", "cp", "chmod"},
}

_MCP_PREFIX_MAP = {
    "mcp____vault_": "mcp:vault",
    "mcp____fact_": "mcp:fact",
    "mcp____backlog_": "mcp:backlog",
    "mcp____autonomy_": "mcp:autonomy",
    "mcp____http_": "mcp:http",
    "mcp____browser_": "mcp:browser",
    "mcp____email_": "mcp:email",
    "mcp____skills_": "mcp:skills",
    "mcp____pipeline_": "mcp:pipeline",
    "mcp____memory_": "mcp:memory",
}

_SIMPLE_TOOL_MAP = {
    "Read": "read",
    "Edit": "edit",
    "Write": "write",
    "Glob": "glob",
    "Grep": "grep",
    "Agent": "agent",
    "ToolSearch": "toolsearch",
    "Skill": "skill",
    "WebFetch": "webfetch",
    "WebSearch": "websearch",
}


def normalize_tool_name(tool: dict) -> str:
    """
    Produce a semantic label for a tool call, used for sequence mining.
    Includes error state: appends ':ERR' if is_error is true.
    """
    name = tool.get("name", "unknown")
    is_error = tool.get("is_error", False)

    # Simple direct mappings
    if name in _SIMPLE_TOOL_MAP:
        label = _SIMPLE_TOOL_MAP[name]
    elif name == "Bash":
        cmd = tool.get("params_summary", {}).get("command", "")
        first_word = cmd.split()[0] if cmd.split() else ""
        label = "bash:other"
        for category, keywords in _BASH_CMD_CATEGORIES.items():
            if first_word in keywords:
                label = category
                break
    elif name.startswith("mcp____"):
        label = "mcp:other"
        for prefix, mapped in _MCP_PREFIX_MAP.items():
            if name.startswith(prefix):
                label = mapped
                break
    else:
        label = name.lower()

    if is_error:
        label += ":ERR"
    return label


# ── Error corroboration (backlog #389) ───────────────────────────────────────
#
# The extractor flags a step as failed only on corroborated signals, and records
# which one in `error_source`. The miner must not re-widen that: it used to read
# `is_error` alone, which is how keyword-matched tool *output* reached skill
# authoring — 188 of 234 flagged steps in the 2026-09-06→08 window had nothing
# behind them, and the `>= 2 pending error candidates` branch in
# `skills/trajectory-skill-mining/SKILL.md` would have emitted the miner's own
# mitigation strings as skills.
#
# Rows written before the extractor fix carry `error_source: "semantic"`, and
# rows written before `error_source` existed carry nothing — both are treated as
# uncorroborated rather than trusted.
CORROBORATED_ERROR_SOURCES = {"protocol", "exit_code"}

# ── What the gate actually rejects (backlog #500) ─────────────────────────────
#
# Keying the gate on `error_source` made it a no-op. That field says which
# channel flagged the step, and there is effectively one channel — the harness
# flag wins every step that has one, so the `elif exit_error` branch in the
# extractor is nearly unreachable. Measured over `_pipeline/trajectories/*.jsonl`
# 2026-09-09→09-17 (9 daily buckets, all written after the flag was wired up):
# 2,330 error steps carried `error_source` = {protocol: 2327, exit_code: 3}, and
# `is_corroborated_error` admitted 2,330 of them. A gate that admits 100% of what
# it is handed is not a gate.
#
# The discriminating field is `failure_class`, which the extractor derives from
# the payload's shape and the signed exit code. `stats.is_error` is a dispatch
# marker: it is true for any non-zero shell exit, and `grep` finding nothing
# (exit 1), `ls` on a missing path (exit 2) and an intentionally failing
# `pytest` are all expected outcomes, not failures worth authoring a skill about.
# Over the flagged messages in `~/lloyd/sessions/*.json` read the same way, that
# class is 1,732 of 3,752 — the largest single bucket, and the one the mining
# chain has already ruled non-failure.
#
# `nonzero_exit` is therefore the only class the gate rejects. Everything else
# stays promotable, because each has an independent reason to be a failure:
# `timeout_or_signal` is a negative exit code (SIGTERM = the real Bash-timeout
# signature, 29 rows), `structured_error` is a body reporting failure as data,
# `harness_block` is the harness refusing or losing the call, and
# `protocol_flagged` is a harness flag with no shape to argue against it.
# Rejecting all flagged steps would repeat, in the other direction, the mistake
# the pre-#389 reading of the same field made.
NON_PROMOTABLE_FAILURE_CLASSES = {"nonzero_exit"}


def is_corroborated_error(step: dict) -> bool:
    """True if this step's failure is a class worth mining.

    `failure_class` decides it when the row carries one. Rows written before
    #500 do not, so the old `error_source` reading survives as the fallback for
    them — and it is why the fallback is narrow: over those rows the same field
    admits every step it is shown, so a re-extraction is what makes the gate
    mean anything (#509).
    """
    # `tools[]` rows carry an explicit `is_error`; `error_tools[]` rows are by
    # construction errors and omit the key.
    if step.get("is_error") is False:
        return False
    fclass = step.get("failure_class")
    if fclass is not None:
        return fclass not in NON_PROMOTABLE_FAILURE_CLASSES
    source = step.get("error_source")
    if source is None:
        # Legacy row with no `error_source`: accept only an explicit non-zero
        # exit state, which is itself a corroborating field.
        return step.get("exit_code") not in (None, 0)
    return source in CORROBORATED_ERROR_SOURCES


SESSION_STORE_DIR = Path.home() / "lloyd" / "sessions"


def set_session_store_dir(path: Path) -> None:
    """Override the session store the class join reads.

    Exists so a test can drive the real command line against a fixture store
    instead of the live 2,800-file `~/lloyd/sessions/`: the class of every row is
    resolved by that join, so without it no subprocess test could state which
    session a row was classified from. `set_trajectory_dir` is its twin.
    """
    global SESSION_STORE_DIR
    SESSION_STORE_DIR = path

# ── Session class (#493, reclassified by #1143) ──────────────────────────────
#
# The gate qualifies a pattern on *distinct sessions*, so the loop's own traffic
# inflates it: machine share of the corpus measured on the stored `platform` field
# is 938 of 1083 sessions (86.6%) across 2026-09-02→12, monotone per day, and
# 10,611 of 11,813 `Sessions Affected` bullets in the 2026-09-12 candidates are
# machine sessions.
#
# Which classes are human-initiated comes from the extractor module, not from a
# second literal here. #493 pinned the two with a test that compared one string;
# #1143 moved the human/non-human split into a set (`HUMAN_CLASSES`), which is the
# thing that actually has to agree across the boundary, and a set copied into two
# files is a set that drifts one member at a time. See the block above
# `SESSION_CLASS` in `extract-trajectories.py` for why `inner_voice` decides
# nothing and why browser-platform chats are admitted.
#
# The class is written by `extract-trajectories.py` and read by this script from
# the session store, per run (see `effective_session_class`).
def _extractor_module():
    """The sibling extractor module, for its session classifier and class sets.

    Loaded by path (hyphenated filename, and `scripts/` is not a package) the same
    way `_verdicts_module()` loads its sibling. One classifier, not two: the
    meaning of `interactive` is clause 2 of #493 as amended by #1143 and cannot
    drift between the writer and the reader of the corpus.
    """
    import importlib.util   # sibling script, loaded by path
    global _EXTRACTOR_MOD
    if _EXTRACTOR_MOD is None:
        path = Path(__file__).resolve().parent / "extract-trajectories.py"
        spec = importlib.util.spec_from_file_location("extract_trajectories", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXTRACTOR_MOD = module
    return _EXTRACTOR_MOD


_EXTRACTOR_MOD: Any = None

INTERACTIVE_CLASS = _extractor_module().INTERACTIVE_CLASS
HUMAN_CLASSES = _extractor_module().HUMAN_CLASSES
UNCODED_CLASS = "uncoded"   # no emitted class and no session JSON to join to
NO_AGENT_ID = "(no agent_id)"   # agent-dropped tally label for a row with no agent


def effective_session_class(traj: dict, cache: dict) -> str:
    """Class the exclusion filters on, joined from the session store.

    The store wins over the row's own `session_class` whenever it has the session:
    the corpus is a derived cache written by `extract-trajectories.py`, so an
    emitted class can only disagree with the store when the classifier changed —
    and then the store's answer is the current one. Trusting a stale `interactive`
    on a session whose JSON says `platform: worker` would reintroduce exactly the
    row #493 is about, so a mis-stamped row is re-classified, not believed. The
    same property is why #1143 needed no backfill: every historical row is
    reclassified through this join at read time, including the 1,411 of the 2,396
    live rows that carry no emitted class at all.

    Every row written before the field existed lacks it (all 1,411 live rows as at
    2026-09-14), and the mining window is 7 days while extraction is incremental —
    so the legacy rows are never rewritten and an exclusion that only understood
    the new field would blank the corpus for a week and emit nothing (the failure
    clause 6 forbids). Hence the join on `session_key`, which is the
    corpus↔store join the acceptance check is written on. A row with neither an
    emitted class nor a session file behind it is `uncoded`: dropped under the
    exclusion and reported, never assumed human.
    """
    key = traj.get("session_key") or ""
    if key in cache:
        return cache[key]
    resolved = None
    if key and "/" not in key and ".." not in key:
        path = SESSION_STORE_DIR / f"{key}.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict):
                # `session_id` falls back to the corpus key: the id shape is half of
                # what the classifier reads (#1143 clause 3), and a row whose JSON
                # carries no id field is still named by the store it came from.
                resolved = _extractor_module().classify_session(data, session_id=key)
    if resolved is None:
        # No session JSON: the emitted class is all that is left, and absence of
        # both is `uncoded`.
        resolved = traj.get("session_class") or UNCODED_CLASS
    cache[key] = resolved
    return resolved


# ── Data loading ─────────────────────────────────────────────────────────────

def load_trajectories(days: int = 7, agent_filter: str = "worker",
                      exclude_machine: bool = True,
                      class_counts: dict | None = None,
                      window: dict | None = None) -> list[dict]:
    """Load trajectory JSONL files with optional filters.

    `exclude_machine` (default True) keeps only the classes in `HUMAN_CLASSES` —
    `interactive` chats and the Chrome extension's `browser` sessions — because the
    frequency gate counts distinct sessions, so the loop's own cadence otherwise
    qualifies its own patterns (#493) while the test traffic that rode the human
    platforms qualified as human work (#1143). Which classes are human is the
    extractor's `HUMAN_CLASSES`, imported not restated: this script and
    `extract-trajectories.py` cannot disagree about what a person's work is.
    `class_counts`, when given a dict, is filled
    with `{"kept": {class: n}, "dropped": {class: n}}` so the caller can report what
    the exclusion removed; the exclusion is never silent. It also carries
    `{"agent-dropped": {agent_id: n}}` — the rows the session class admitted and
    `agent_filter` then rejected, the half of the selection that used to be silent.
    `window`, when given a dict, is filled with `{"files": n, "rows": n}`: how many
    dated JSONL buckets fell in the window and how many rows they held. That is what
    lets a caller tell an empty window from a filter that selected nothing (#998) —
    without it the two empties print identically, and a mis-set filter exits 0 while
    `write_index` rewrites the live candidate index to `Total candidates: 0`.
    """
    trajectories = []
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    class_cache: dict[str, str] = {}

    if not TRAJECTORY_DIR.exists():
        print(f"Warning: Trajectory directory not found: {TRAJECTORY_DIR}")
        if window is not None:
            window.update(files=0, rows=0)
        return trajectories

    def tally(bucket: str, cls: str) -> None:
        if class_counts is not None:
            class_counts.setdefault(bucket, {})
            class_counts[bucket][cls] = class_counts[bucket].get(cls, 0) + 1

    if window is not None:
        window.update(files=0, rows=0)

    for jsonl_file in sorted(TRAJECTORY_DIR.glob("*.jsonl")):
        # Skip non-date files
        if not re.match(r'^\d{4}-\d{2}-\d{2}\.jsonl$', jsonl_file.name):
            continue
        
        # Check date filter — compare by date only so that e.g. 2026-04-06.jsonl
        # is included when --days 1 is used at any time on 2026-04-07.
        try:
            file_date = datetime.strptime(jsonl_file.stem, "%Y-%m-%d").date()
            if file_date < cutoff.date():
                continue
        except ValueError:
            continue

        if window is not None:
            window["files"] += 1

        try:
            with open(jsonl_file, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        traj = json.loads(line)
                        # Counted before either selector runs, so `rows` answers
                        # "was there anything in the window to select from" and
                        # cannot read as 0 for the case it exists to detect —
                        # machine-class rows that the exclusion removes before the
                        # agent filter ever sees them (#998).
                        if window is not None:
                            window["rows"] += 1
                        # Session-class exclusion (#493). `agent_id` cannot do
                        # this work: the extractor derives it from the filename
                        # and 0 of the live session files carry its one prefix, so
                        # every row reads `agent_id: "lloyd"` — 1,306 of 1,306 in
                        # the seven newest buckets, measured 2026-09-14 — and
                        # `--agent worker` selects nothing (#494).
                        if exclude_machine:
                            cls = effective_session_class(traj, class_cache)
                            # Stamped with the class the exclusion actually filtered
                            # on, so `print_stats` cannot contradict the tally above
                            # it: a row admitted because the store says it is
                            # interactive would otherwise print as `uncoded`, and a
                            # row re-classified from the store would print as its
                            # stale emitted value.
                            traj["session_class"] = cls
                            if cls in HUMAN_CLASSES:
                                tally("kept", cls)
                            else:
                                tally("dropped", cls)
                                continue
                        else:
                            tally("kept", traj.get("session_class") or UNCODED_CLASS)
                        # Apply agent filter
                        # agent_id values from extract-trajectories.py:
                        #   "autonomy" → worker/autonomy sessions
                        #   "lloyd"    → interactive/main sessions
                        if agent_filter != "all":
                            aid = traj.get("agent_id", "")
                            if agent_filter in ("worker", "autonomy"):
                                selected = aid in ("worker", "autonomy")
                            elif agent_filter in ("main", "lloyd"):
                                selected = aid in ("main", "lloyd")
                            else:
                                selected = aid == agent_filter
                            if not selected:
                                # Counted, not swallowed. This is the selection the
                                # documented `--agent worker` default fails on every
                                # live row — 0 of 1,306 carry `agent_id` worker or
                                # autonomy (#494) — and the number a non-zero exit
                                # has to name so a zero run says which filter did it
                                # instead of printing a SUMMARY (#998).
                                tally("agent-dropped", aid or NO_AGENT_ID)
                                continue
                        trajectories.append(traj)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        except Exception as e:
            print(f"Warning: Could not read {jsonl_file}: {e}")
            continue
    
    return trajectories


# ── Pattern mining ───────────────────────────────────────────────────────────

# How many instances of one pattern key are printed as examples. The number is a
# *sample*, not the key's evidence, which is why `write_candidate_file` emits
# `examples_shown:` beside the key's own denominator (#1231 clause 2). The caps
# are unchanged from the first-N-wins code they replace — the change is which
# instances they select, not how many.
EXAMPLE_CAP_ERROR = 5
EXAMPLE_CAP_SUCCESS = 3
EXAMPLE_CAP_SEQUENCE = 3


def pool_offer(pool: list, cap: int, entry, order_key) -> None:
    """Keep the newest `cap` instances of a pattern, whatever the scan order.

    `pool` holds `(order_key, entry)` pairs kept sorted oldest-first, so an
    instance arriving past the cap evicts the *oldest* held one and the sample
    rolls forward. The old shape gated the append on the pool's own length and
    stopped appending once N was reached, so the printed payload was the first N
    instances in scan order forever. (This prose deliberately does not quote that
    old statement: the item proves the caps are gone by grepping the file for it,
    and a docstring match would read as a cap left in place.) Scan order on the
    live corpus is
    chronological ascending (`load_trajectories` iterates a sorted glob), so
    first-N-wins meant the *oldest* N in the window: measured over the 901 keys
    with 2+ dated snapshots in `_pipeline/skills/candidates/`, 625 carried a
    byte-identical evidence section first-to-last while `sessions` grew up to
    22.2x, and `seq-2-read-write` held one sha across 5 snapshots from 38
    sessions to 275 (#1231 clause 1).

    `order_key` must be unique within one mining run — every caller passes
    `(date_str, <per-run counter>)` — because the pool is sorted by tuple
    comparison and a tie would compare the example dicts, which are not
    orderable.
    """
    if len(pool) < cap:
        insort(pool, (order_key, entry))
        return
    if order_key > pool[0][0]:
        pool.pop(0)
        insort(pool, (order_key, entry))


def pool_sample(pool: list) -> list:
    """The held instances, oldest-first — the same order the printed examples
    have always had, so only the *selection* changes, not the reading order."""
    return [entry for _order_key, entry in pool]


def mine_error_patterns(trajectories: list[dict], threshold: int = 2) -> list[dict]:
    """
    Mine error patterns from trajectories.
    Groups errors by (tool_name, error_category, params_signature).
    Returns patterns that appear in >= threshold distinct sessions.
    """
    # Structure: {(tool_name, error_category, params_sig): {session_keys: set, examples: pool, dates: set}}
    # `examples` is a `pool_offer` pool of (order_key, example) pairs, unwrapped
    # into a plain list of examples only when the pattern qualifies (#1231).
    pattern_data = defaultdict(lambda: {
        "sessions": set(),
        "examples": [],
        "dates": set(),
        "total_calls": 0
    })
    
    for traj in trajectories:
        session_key = traj.get("session_key", "unknown")
        timestamp = traj.get("timestamp", "")
        
        # Extract date from timestamp
        try:
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        
        # Process error_tools
        for error_tool in traj.get("error_tools", []):
            if not is_corroborated_error(error_tool):
                continue
            tool_name = error_tool.get("name", "unknown")
            error_type = error_tool.get("error_type", "logic")
            params_summary = error_tool.get("params_summary", {})
            # An un-keyable call keeps the error: error candidates are filed on
            # (tool, error_type) and the signature only splits variants, so
            # dropping it here would delete failures over a grouping nuisance.
            params_sig = normalize_params_signature(params_summary) or "generic"
            
            key = (tool_name, error_type, params_sig)
            pattern_data[key]["sessions"].add(session_key)
            pattern_data[key]["dates"].add(date_str)
            pattern_data[key]["total_calls"] += 1
            
            # Sample the newest 5 instances instead of the first 5 seen.
            # `total_calls` just incremented, so it is this offer's position in
            # the key's scan order and makes the order key unique (#1231).
            example = {
                "session_key": session_key,
                "date": date_str,
                "tool": tool_name,
                "error_type": error_type,
                # The class travels into the candidate's examples so the
                # reader can see what kind of failure they are grading, not
                # only that the harness flagged it (#500).
                "failure_class": error_tool.get("failure_class"),
                "params_summary": params_summary,
                "sequence": error_tool.get("sequence", 0)
            }
            pool_offer(pattern_data[key]["examples"], EXAMPLE_CAP_ERROR,
                       example, (date_str, pattern_data[key]["total_calls"]))
    
    # Filter by threshold
    qualifying_patterns = []
    for key, data in pattern_data.items():
        if len(data["sessions"]) >= threshold:
            tool_name, error_type, params_sig = key
            qualifying_patterns.append({
                "type": "error",
                "tool_name": tool_name,
                "error_type": error_type,
                "params_signature": params_sig,
                "sessions": data["sessions"],
                "examples": pool_sample(data["examples"]),
                "dates": data["dates"],
                "total_calls": data["total_calls"],
                "first_seen": min(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                "last_seen": max(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            })
    
    return qualifying_patterns


def mine_success_patterns(trajectories: list[dict], threshold: int = 2) -> list[dict]:
    """
    Mine successful tool sequences that repeat across sessions.
    Groups by (tool_name, params_signature) for non-error calls.
    Returns patterns that appear in >= threshold distinct sessions.
    """
    # Structure: {(tool_name, params_sig): {session_keys: set, examples: list, dates: set, total: int}}
    pattern_data = defaultdict(lambda: {
        "sessions": set(),
        "examples": [],
        "dates": set(),
        "total_calls": 0,
        "error_count": 0
    })
    
    for traj in trajectories:
        session_key = traj.get("session_key", "unknown")
        timestamp = traj.get("timestamp", "")
        
        # Extract date from timestamp
        try:
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        
        # Process all tools (both success and error)
        for tool in traj.get("tools", []):
            tool_name = tool.get("name", "unknown")
            is_error = tool.get("is_error", False)
            params_summary = tool.get("params_summary", {})
            params_sig = normalize_params_signature(params_summary) if params_summary else "generic"
            if params_sig == "":
                # Nothing identifiable about this call — a command that was
                # scrubbed before #391 preserved the verb. It used to land in
                # one shared bucket keyed on the scrub placeholder, which is how
                # an unrelated 625-call union got mined as a single skill. Drop
                # it: a signature-mining input with no key is not evidence.
                continue
            
            key = (tool_name, params_sig)
            pattern_data[key]["sessions"].add(session_key)
            pattern_data[key]["dates"].add(date_str)
            pattern_data[key]["total_calls"] += 1
            
            if is_error and is_corroborated_error(tool):
                pattern_data[key]["error_count"] += 1
            
            # Sample the newest 3 instances instead of the first 3 seen;
            # `total_calls` is this offer's position in the key's scan order
            # (#1231).
            example = {
                "session_key": session_key,
                "date": date_str,
                "tool": tool_name,
                "params_summary": params_summary,
                "result_summary": tool.get("result_summary", ""),
                "is_error": is_error,
                "sequence": tool.get("sequence", 0)
            }
            pool_offer(pattern_data[key]["examples"], EXAMPLE_CAP_SUCCESS,
                       example, (date_str, pattern_data[key]["total_calls"]))
    
    # Filter by threshold and focus on high-frequency patterns
    qualifying_patterns = []
    for key, data in pattern_data.items():
        if len(data["sessions"]) >= threshold and data["total_calls"] >= threshold:
            tool_name, params_sig = key
            error_rate = data["error_count"] / data["total_calls"] if data["total_calls"] > 0 else 0
            
            # Only include if it has some success rate (not 100% errors)
            if error_rate < 0.9:
                qualifying_patterns.append({
                    "type": "success",
                    "tool_name": tool_name,
                    "params_signature": params_sig,
                    "sessions": data["sessions"],
                    "examples": pool_sample(data["examples"]),
                    "dates": data["dates"],
                    "total_calls": data["total_calls"],
                    "error_count": data["error_count"],
                    "error_rate": error_rate,
                    "first_seen": min(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                    "last_seen": max(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                })
    
    # Sort by frequency
    qualifying_patterns.sort(key=lambda x: -x["total_calls"])
    return qualifying_patterns


ERR_SUFFIX = ":ERR"

# The shortest argument value that can name an object, and the two characters
# that say it does: a path, a URL, a filename or a dotted id carries one of
# them. `limit: 200`, `offset: 0` and `output_mode: count` name no object at all,
# and matching on those would rebuild the adjacency bug out of argument values.
MIN_REFERENCE_LEN = 5
REFERENCE_MARKS = ("/", ".")


def _reference_values(params_summary) -> set[str]:
    """The argument values of one call that name the *object* it touched.

    Values are compared as whole strings across the two calls, not by argument
    name — a failed `Edit` names its target `path` or `file_path` depending on
    which extractor wrote the row, and the retry names it with whatever its own
    tool calls it. The identity is the value. Scrubbed values
    (`[truncated: N chars]`, `[MASKED]`) are the same non-evidence #391 removed
    from the keys and are skipped: a placeholder equals every placeholder.
    """
    if not isinstance(params_summary, dict):
        return set()
    values: set[str] = set()
    for value in params_summary.values():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if item is None or isinstance(item, (bool, dict)):
                continue
            text = str(item).strip()
            if len(text) < MIN_REFERENCE_LEN or text.startswith(KEY_ARTIFACT_PREFIX):
                continue
            if not any(mark in text for mark in REFERENCE_MARKS):
                continue
            values.add(text)
    return values


def ngram_shows_recovery(labels: list[str] | tuple[str, ...],
                         tools: list[dict]) -> bool:
    """Whether one observed instance of an n-gram contains a recovery, not just a
    next step (backlog #1327).

    A step whose label ends in `:ERR` counts as recovered only when a *later*
    step in the same window addresses that failure, in one of the two ways the
    item names:

    * it **re-attempts** the failed call — its base label is the failed step's
      base label, so the same tool ran again without the error flag (a retried
      `mkdir`, a second `Read`); or
    * it **references the failed target** — it carries an argument value equal to
      one of the failed call's object-naming values (the same path, URL or
      command), which is how a failed `Edit` followed by a `Read` of that same
      file differs from a failed `Edit` followed by a `Read` of another one.

    Anything else leaves the flag false. Bare adjacency — `:ERR` followed by any
    non-`:ERR` step — used to set it, and that is not recovery: of the five
    candidate keys consolidation hand-adjudicated from the keys this flag kept on
    2026-09-21, four had successors that were unrelated next commands (`date -u …`
    after a failed grep, a fresh `ls` after a traceback) and in one the "error"
    was legitimate stdout on a non-zero exit. An n-gram of one step, or one with
    no `:ERR` step, has nothing to recover from and returns False.
    """
    for idx, label in enumerate(labels):
        if not label.endswith(ERR_SUFFIX):
            continue
        base = label[:-len(ERR_SUFFIX)]
        failed_refs = _reference_values(
            tools[idx].get("params_summary") if idx < len(tools) else None)
        for later in range(idx + 1, len(labels)):
            later_label = labels[later]
            if later_label.endswith(ERR_SUFFIX):
                continue
            if later_label == base:
                return True
            if idx < len(tools) and later < len(tools) and failed_refs:
                if failed_refs & _reference_values(
                        tools[later].get("params_summary")):
                    return True
    return False


def _suffix_sessions(ngram: tuple[str, ...],
                     pattern_data: dict[tuple[str, ...], dict]) -> set[str]:
    """Sessions a strictly shorter *suffix* of this n-gram already counted.

    `mine_sequence_patterns` walks `for n in (2, 3)` over one collapsed label
    stream per session, so a 3-gram at position *i* always feeds its suffix
    bigram at *i+1* in the same session: a `seq-3` key's session set is a subset
    of its suffix bigram's by construction, and its sessions are the same events
    counted at a second window size. `seq-3-bash-fs-bash-fs-err-bash-other`
    shared all 17 of its sessions with the 20 that
    `seq-2-bash-fs-err-bash-other` reported. Measured over
    `_pipeline/skills/candidates/` on 2026-09-21: 626 `seq-3` keys, 594 with a
    `seq-2` suffix sibling, 570 of those pairs fully contained. Borrowed here, so
    a windowed key cannot clear the emission gate on a session its own suffix
    already billed.
    """
    borrowed: set[str] = set()
    for size in range(2, len(ngram)):
        suffix = tuple(ngram[-size:])
        sibling = pattern_data.get(suffix)
        if sibling is not None:
            borrowed |= sibling["sessions"]
    return borrowed


def mine_sequence_patterns(trajectories: list[dict], threshold: int = 2) -> list[dict]:
    """
    Mine repeating tool-call sequences (bigrams and trigrams) across sessions.
    Normalizes tool names, collapses consecutive duplicates, then extracts
    n-grams. Returns patterns appearing in >= threshold distinct sessions.

    Two rules decide what survives the threshold, both from backlog #1327:

    * `has_error_recovery` is set per observed instance by
      `ngram_shows_recovery()`, so a key is flagged true only when at least one
      session actually re-attempted the failed call or touched its target again
      — not because an `:ERR` label happened to sit next to a non-`:ERR` label;
    * a key is qualified on its **own** sessions: those not already counted by a
      strictly shorter suffix key (`_suffix_sessions`). A windowed n-gram reports
      `sessions` as that own count, with the full observed total in
      `total_sessions` and what the suffix already billed in `borrowed_sessions`.
    """
    # {ngram_tuple: {sessions, examples, dates}}
    pattern_data: dict[tuple[str, ...], dict] = defaultdict(lambda: {
        "sessions": set(),
        "examples": [],
        "dates": set(),
    })

    for traj in trajectories:
        session_key = traj.get("session_key", "unknown")
        timestamp = traj.get("timestamp", "")

        try:
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        tools = traj.get("tools", [])
        if len(tools) < 2:
            continue

        # Sort by sequence number
        sorted_tools = sorted(tools, key=lambda t: t.get("sequence", 0))

        # Normalize labels
        labels = [normalize_tool_name(t) for t in sorted_tools]

        # Collapse consecutive identical labels (keep first tool of each run)
        collapsed_labels: list[str] = []
        collapsed_tools: list[dict] = []
        for label, tool in zip(labels, sorted_tools):
            if not collapsed_labels or label != collapsed_labels[-1]:
                collapsed_labels.append(label)
                collapsed_tools.append(tool)

        # Track which ngrams we've already counted for this session (dedup within session)
        seen_in_session: set[tuple[str, ...]] = set()

        for n in (2, 3):
            for i in range(len(collapsed_labels) - n + 1):
                ngram = tuple(collapsed_labels[i : i + n])
                if ngram in seen_in_session:
                    continue
                seen_in_session.add(ngram)

                pd = pattern_data[ngram]
                # A sequence key has no instance counter distinct from its
                # sessions: `seen_in_session` above dedups the n-gram within one
                # row, and this keeps one offer per session even if the same
                # `session_key` arrives in two rows. So offers == len(sessions)
                # exactly, and `examples_shown` is comparable to `sessions`
                # without inventing an instance tally (#1231).
                first_from_session = session_key not in pd["sessions"]
                pd["sessions"].add(session_key)
                pd["dates"].add(date_str)
                # Recovery is a property of an observed instance, not of the
                # label shape: the same `edit:ERR → read` n-gram is a recovery
                # when the `Read` opens the file the `Edit` failed on and is not
                # when it opens another one. OR'd across every instance the key
                # was seen in, so the flag says "at least one session in this
                # set recovered" and never "the next step happened to exist".
                if not pd.get("has_error_recovery"):
                    window_tools = collapsed_tools[i: i + n]
                    pd["has_error_recovery"] = ngram_shows_recovery(
                        collapsed_labels[i: i + n], window_tools)

                # Sample up to 3 concrete examples, newest-first rather than
                # first-wins.
                if not first_from_session:
                    continue
                example_steps = []
                for j in range(n):
                    t = collapsed_tools[i + j]
                    example_steps.append({
                        "tool": t.get("name", "unknown"),
                        "label": collapsed_labels[i + j],
                        "params_summary": t.get("params_summary", {}),
                        "result_summary": t.get("result_summary", ""),
                        "is_error": t.get("is_error", False),
                    })
                pd["offers"] = pd.get("offers", 0) + 1
                pool_offer(pd["examples"], EXAMPLE_CAP_SEQUENCE, {
                    "session_key": session_key,
                    "date": date_str,
                    "steps": example_steps,
                }, (date_str, pd["offers"]))

    # Boring sequences that every agent session produces — filter these out
    # to focus on genuinely interesting multi-step patterns
    BORING_LABELS = {"read", "glob", "grep", "bash:explore", "bash:other", "toolsearch"}

    # Filter by threshold
    qualifying: list[dict] = []
    for ngram, data in pattern_data.items():
        # The threshold is on the key's OWN sessions. A windowed n-gram's set is
        # a subset of its suffix's (`_suffix_sessions`), so counting it again at
        # its own window size is what let `seq-3-bash-fs-bash-fs-err-bash-other`
        # clear `sessions >= 3` on 17 sessions every one of which its suffix
        # `seq-2-bash-fs-err-bash-other` had already billed.
        borrowed = _suffix_sessions(ngram, pattern_data)
        own_sessions = data["sessions"] - borrowed
        if len(own_sessions) < threshold:
            continue

        has_error_recovery = bool(data.get("has_error_recovery"))

        # Skip sequences composed entirely of boring labels (no :ERR, no MCP, no edit/write)
        base_labels = {lbl.split(":ERR")[0] for lbl in ngram}
        if base_labels.issubset(BORING_LABELS) and not has_error_recovery:
            continue

        qualifying.append({
            "type": "sequence",
            "sequence": ngram,
            "sequence_str": " → ".join(ngram),
            "ngram_size": len(ngram),
            "sessions": own_sessions,
            "total_sessions": len(data["sessions"]),
            "borrowed_sessions": len(data["sessions"]) - len(own_sessions),
            "examples": pool_sample(data["examples"]),
            "dates": data["dates"],
            "has_error_recovery": has_error_recovery,
            "first_seen": min(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "last_seen": max(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        })

    # Sort by session count descending, then by ngram size descending
    qualifying.sort(key=lambda x: (-len(x["sessions"]), -x["ngram_size"]))
    return qualifying


# ── Output generation ────────────────────────────────────────────────────────

def generate_mitigation(tool_name: str, error_type: str) -> str:
    """Generate a suggested mitigation based on tool and error type."""
    mitigations = {
        ("run_bash", "permission"): "Check file ownership and permissions before executing commands. Use sudo with explicit path validation when elevated privileges are needed.",
        ("run_bash", "not_found"): "Verify file/directory paths exist before operations. Consider adding existence checks or using absolute paths.",
        ("run_bash", "timeout"): "Review command complexity and consider breaking into smaller operations. Add progress indicators for long-running tasks.",
        ("run_bash", "network"): "Check network connectivity and DNS resolution. Consider adding retry logic with exponential backoff.",
        ("run_bash", "validation"): "Validate input parameters before command execution. Use schema validation for complex arguments.",
        ("run_bash", "resource"): "Monitor system resources (disk, memory) before heavy operations. Consider cleanup strategies.",
        ("file_read", "permission"): "Check file ownership and ensure read permissions. Consider using absolute paths and verifying access before reading.",
        ("file_read", "not_found"): "Verify file existence before attempting to read. Consider graceful fallback for missing files.",
        ("file_write", "permission"): "Check directory write permissions and disk space. Consider using temp files with atomic moves.",
        ("file_write", "not_found"): "Ensure parent directories exist before writing. Use mkdir -p or equivalent for path creation.",
        ("http_fetch", "network"): "Add retry logic with exponential backoff. Consider timeout configurations and connection pooling.",
        ("http_fetch", "timeout"): "Increase timeout values for large responses. Consider streaming for large payloads.",
        ("http_request", "network"): "Verify endpoint availability and network configuration. Add health checks before requests.",
        ("http_request", "timeout"): "Review timeout settings based on expected response times. Consider async operations for long requests.",
    }
    
    key = (tool_name, error_type)
    if key in mitigations:
        return mitigations[key]
    
    # Default mitigations by error type
    default_by_type = {
        "permission": "Check file/directory permissions and ownership. Consider using appropriate privilege escalation when needed.",
        "not_found": "Verify paths and existence before operations. Add defensive checks for missing resources.",
        "timeout": "Review operation complexity and consider breaking into smaller steps. Add progress tracking.",
        "network": "Implement retry logic with exponential backoff. Add connection health checks.",
        "validation": "Add input validation before processing. Use schema validation for complex structures.",
        "resource": "Monitor system resources and implement cleanup strategies. Consider batch processing for large operations.",
        "logic": "Review error handling logic. Add more specific error checking and fallback behavior.",
    }
    
    return default_by_type.get(error_type, "Review error context and add appropriate handling for this scenario.")


def generate_title(pattern: dict) -> str:
    """Generate a descriptive title for a pattern."""
    if pattern["type"] == "error":
        tool_name = pattern["tool_name"]
        error_type = pattern["error_type"]
        return f"Handle {tool_name} {error_type} errors"
    elif pattern["type"] == "sequence":
        return f"Sequence: {pattern['sequence_str']}"
    else:
        tool_name = pattern["tool_name"]
        return f"Pattern: {tool_name} usage"


# ── Verdict ledger gate (#530) ───────────────────────────────────────────────
#
# Placed here rather than at the head of the file, and the ledger module imported
# lazily inside a function, for two separate reasons. Function-local because
# `skill_verdicts.py` is a sibling script, not a package member: it resolves when the
# miner runs as `python3 scripts/mine-trajectories.py` and also when a test loads this
# file with importlib, where `scripts/` is not on sys.path. If the import fails the
# error propagates — a silently-absent gate would report "no verdicts" and re-mint
# every rejected pattern, which is the exact defect this closes.
#
# The pre-existing `normalize_tool_name` redefinition above is a pyflakes finding whose
# message embeds its own line number, and the gate's normaliser strips only the
# `path:line:col:` prefix — so anything inserted higher in this file reports that
# finding as new and fails the static rung on a change that introduced nothing. Filed
# separately; nothing here is above it.

def _verdicts_module():
    """The sibling `skill_verdicts` module, importable from either call style."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import skill_verdicts
    return skill_verdicts


def candidate_pattern_key(pattern: dict) -> str:
    """The key a mined pattern is adjudicated under.

    It must equal the `pattern:` frontmatter field written below, because that is what
    `skill_verdicts.py` reads back out of a candidate when the runbook asks "has this
    already been decided?". The ledger and the corpus join on this string and nothing
    else, so it is derived in exactly one place. Error patterns key on
    `(tool, error_type)` — `Bash/timeout`, `Edit/not_found` — which is the shape the
    seven rejected 09-06 candidates actually carry. A sequence keys on the n-gram
    itself through `slug_for`, so a long n-gram carries a disambiguator instead of
    being silently cut at 50 slug characters — the two shapes that shared key
    `seq-5-backlog-write-task-bash-fs-automod-gate-wait-autom` were different
    patterns joined onto one ledger row (#1131).
    """
    if pattern["type"] == "error":
        return f"{pattern['tool_name']}/{pattern['error_type']}"
    if pattern["type"] == "sequence":
        return sequence_pattern_key(pattern["ngram_size"], pattern["sequence_str"])
    return f"{pattern['tool_name']}/{pattern['params_signature']}"


def candidate_filename(pattern: dict, today: str | None = None) -> str:
    """The file one pattern writes to: `candidate-{slug_for(key)}-{today}.md`.

    `slug_for`, not `slugify`: the key of a sequence that hit the cap is longer than
    the cap itself, so a plain `slugify` here re-cut it at 50 and put the two 5-grams
    back onto one filename — the disambiguator arriving at byte 51 onward, exactly
    where the cut lands. Sluging the key through the same cap-scoped rule appends a
    hash of the *key*, which is injective on the pattern; the two mechanisms together
    are what make one pattern own one file (#1131, clause 1).

    Split out of `write_candidate_file` so `emit_candidates` can ask the collision
    question over every class **before** its first write. Deriving the name twice by
    two code paths is the thing to avoid, so the writer calls this too and there is
    one rule for the name (#515, clause 1).
    """
    today = today or datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    return f"candidate-{slug_for(candidate_pattern_key(pattern))}-{today}.md"


def sequence_pattern_key(ngram_size: int, sequence_str: str) -> str:
    """The key for one mined n-gram: `seq-{n}-{slug}`, disambiguator included.

    `slug_for` sees the **whole** key text, so the cap is measured on the key and
    not on the n-gram string. That is the only measurement that means anything
    downstream: the key is what `write_candidate_file` slugs for the filename and
    what the verdict ledger stores, and `seq-3-` already costs 6 characters — an
    n-gram whose own slug is 49 characters is 5 bytes past the cut once the prefix
    is on it, so a cap measured on the n-gram alone would keep minting
    under-the-cap-looking keys that the filename step then truncated. One string,
    one cap, one rule at both levels.

    Split out from `candidate_pattern_key` so the rule lives in exactly one place
    and a caller that has the n-gram but no pattern dict — a test, or a ledger
    lookup assembled from a `sequence_str` — can derive the same string. Below the
    cap the function is the identity `seq-{n}-{slugify(sequence_str)}`, which is
    what keeps every `seq-*` row already in the ledger reachable (#1131 clause 4).
    """
    return slug_for(f"seq-{ngram_size}-{sequence_str}")


# How many error examples survive into one candidate file. A merged candidate
# carries the evidence of every signature bucket behind its key, so the cap is
# spent one bucket at a time rather than taken from whichever bucket the mined
# dict happened to yield first — that arbitrary choice is #515's defect.
MAX_MERGED_EXAMPLES = 5


def _bucket_row(pattern: dict) -> dict:
    """One mined signature bucket, in the shape a merged candidate reports it."""
    return {
        "params_signature": str(pattern.get("params_signature") or "generic"),
        "occurrences": int(pattern.get("total_calls") or 0),
        "sessions": len(pattern.get("sessions") or ()),
        "first_seen": pattern.get("first_seen", ""),
        "last_seen": pattern.get("last_seen", ""),
    }


def _merged_examples(buckets: list[dict], cap: int = MAX_MERGED_EXAMPLES) -> list[dict]:
    """Up to `cap` examples, taken one per signature bucket on each pass.

    The buckets arrive in sorted order, so reversing the mined pattern list cannot
    change which examples reach the file — the byte-identity #515 clause 3 asks for.
    """
    chosen: list[dict] = []
    for rank in range(cap):
        for bucket in buckets:
            examples = bucket.get("examples") or []
            if rank < len(examples) and len(chosen) < cap:
                chosen.append(examples[rank])
    return chosen


def merge_error_patterns(patterns: list[dict]) -> list[dict]:
    """One emission unit per candidate key, carrying every bucket behind that key.

    `mine_error_patterns` groups on `(tool_name, error_type, params_signature)`
    while `candidate_pattern_key` deliberately drops the signature, because
    `Tool/error_type` is the shape the verdict ledger adjudicates. Two things
    follow from that mismatch, both measured over the live corpus on 2026-09-14:
    73 mined error patterns collapse to 25 keys, so a per-pattern writer puts 73
    writes at 25 paths and keeps only the last one (`Bash/logic`: 19 buckets, 531
    occurrences over 340 sessions, filed with one bucket's numbers), and one
    verdict gates every bucket behind its key.

    Merging at emission respects both halves: mining keeps its finer telemetry,
    the ledger keeps its coarse join, and each key gets one file whose
    `occurrences:` and `sessions:` are the sum and the union over its buckets,
    with the per-bucket breakdown printed inside the file so a reader can see how
    many distinct mined patterns one verdict is standing in front of (#515).

    Idempotent: a group of one is copied through with its own one-row breakdown,
    so feeding a merged list back in cannot rewrite an already-merged candidate.
    """
    groups: dict[str, list[dict]] = {}
    rest: list[dict] = []
    for pattern in patterns:
        if pattern.get("type") == "error":
            groups.setdefault(candidate_pattern_key(pattern), []).append(pattern)
        else:
            rest.append(pattern)

    def single(pattern: dict) -> dict:
        """A key with one bucket behind it: same numbers, breakdown filled in.

        A pattern that arrived already merged keeps its own breakdown rows rather
        than being re-described as one bucket, which is what makes this function
        idempotent.
        """
        if pattern.get("merged_buckets"):
            return dict(pattern)
        return dict(pattern, merged_from=1, merged_buckets=[_bucket_row(pattern)])

    def combined(buckets: list[dict]) -> dict:
        head = dict(buckets[0])
        sessions: set = set()
        dates: set = set()
        for bucket in buckets:
            sessions |= set(bucket.get("sessions") or ())
            dates |= set(bucket.get("dates") or ())
        head.update(
            total_calls=sum(int(b.get("total_calls") or 0) for b in buckets),
            sessions=sessions,
            dates=dates,
            examples=_merged_examples(buckets),
            params_signature="+".join(sorted(
                {str(b.get("params_signature") or "generic") for b in buckets})),
            merged_from=len(buckets),
            merged_buckets=[_bucket_row(b) for b in buckets],
        )
        if dates:
            head["first_seen"] = min(dates)
            head["last_seen"] = max(dates)
        return head

    merged = []
    for key in sorted(groups):
        buckets = sorted(groups[key],
                         key=lambda b: (str(b.get("params_signature") or "generic"),
                                        int(b.get("total_calls") or 0)))
        if len(buckets) == 1:
            merged.append(single(buckets[0]))
        else:
            merged.append(combined(buckets))
    return merged + rest


def merge_note(pattern: dict) -> str:
    """What one emitted file stands for, in the units the run actually merged.

    Every emitted pattern gets a statement, not only the merged ones: the count is
    derived from `merged_buckets`, which only `merge_error_patterns` creates. A
    pattern with no buckets of its own — a success or sequence candidate, or an error
    one that bypassed the merge — says so instead of printing a bare `1` the run has
    no way to know.
    """
    buckets = pattern.get("merged_buckets")
    if buckets is None:
        return "not signature-keyed (non-error candidate)"
    if len(buckets) > 1:
        return f"{len(buckets)} mined patterns merged onto this key"
    return "1 mined pattern, nothing merged"


def verdict_for(pattern: dict, store: str | Path | None = None) -> dict | None:
    """The binding terminal verdict blocking this pattern, or None if it is free.

    An absent ledger file returns None for every key: that is the honest empty state,
    and the runbook has to print `skipped_by_verdict: 0` out loud rather than have the
    gate pretend it saw something.

    An error unit with MORE THAN ONE signature bucket behind its key (#515) skips the
    >10x growth trigger. Its `total_calls` counts every bucket behind the key while
    the ledger's `occurrences_at_decision` was recorded from ONE bucket's file, since
    #530 seeded the ledger before the buckets were ever merged. Comparing the two is a
    units error, and it would reopen exactly the multi-bucket keys on the first run
    after the merge — re-adjudicating a stack of patterns nobody asked to reopen,
    which is the decision Alan reserved on #515 (whether the existing coarse rows
    should be re-adjudicated per signature). Terminality and the 60-day expiry still
    bind, and the suppression set is therefore the same one as before the merge. The
    candidate body says this out loud.

    A key with one bucket behind it keeps the growth rule: there `total_calls` is that
    one bucket's count, so it and the stored baseline have the same denominator and
    zeroing it would suppress a key that is genuinely entitled to reopen.
    """
    occurrences = int(pattern.get("total_calls") or 0)
    if len(pattern.get("merged_buckets") or ()) > 1:
        occurrences = 0
    return _verdicts_module().terminal_verdict(
        candidate_pattern_key(pattern),
        store=store,
        occurrences=occurrences,
    )


def status_block(pattern: dict, store: str | Path | None = None) -> tuple[str, str, dict | None]:
    """(`status` value, extra frontmatter, verdict) for one candidate.

    A pattern carrying a terminal verdict is still written — its evidence is telemetry
    and the corpus is the audit trail — but never as `pending_review`. It arrives as
    `superseded_by_verdict`, so a re-mined snapshot cannot present already-rejected
    content as a fresh decision, and the reason that rejected it travels with the file
    instead of being re-derived by hand the next night.
    """
    verdict = verdict_for(pattern, store=store)
    if not verdict:
        return "pending_review", "", None
    reason = " ".join(str(verdict.get("reason", "")).split())
    front = (
        f"\nverdict: {verdict.get('verdict', '')}"
        f"\nverdict_decided_at: {verdict.get('decided_at', '')}"
        f"\nverdict_source: {verdict.get('source_candidate', '')}"
        f"\nverdict_evidence_cmd: {verdict.get('evidence_cmd', '')}"
        f"\nverdict_reason: {reason}"
    )
    return "superseded_by_verdict", front, verdict


def superseded_pattern_keys(patterns: list[dict], store: str | Path | None = None) -> list[str]:
    """Keys the ledger blocks, for the run summary.

    Same lookup `write_candidate_file` performs, exposed so the nightly report can state
    a number and a reason per key. #391 records that consolidation "returns zero every
    night" with no stated reason; an unexplained zero reads as a broken job, and a
    reported one reads as a decision.
    """
    keys = []
    for pattern in patterns:
        if verdict_for(pattern, store=store):
            keys.append(candidate_pattern_key(pattern))
    return keys


def write_candidate_file(pattern: dict, output_dir: Path, verdict_store: str | Path | None = None) -> str | None:
    """Write a single candidate markdown file. Returns the file path, or None
    if the pattern is not a skill candidate at all (`is_emittable`). The check
    is here as well as in `emit_candidates` so no caller can re-open the hole.

    `verdict_store` additionally decides the `status` line, not whether the file is
    written: a pattern whose key carries a terminal verdict is emitted as
    `superseded_by_verdict` rather than `pending_review` (#530).
    """
    if not is_emittable(pattern):
        return None

    # The one caller creates the dir; creating it here too means a caller that forgets
    # gets a candidate file, not a FileNotFoundError halfway through a nightly run.
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern_slug = candidate_pattern_key(pattern)
    status_value, verdict_fm, _verdict = status_block(pattern, store=verdict_store)

    # Derived through the one function that knows the rule, so the name this writer
    # produces is necessarily the name `emit_candidates` pre-flight-checked.
    filename = candidate_filename(pattern)
    filepath = output_dir / filename

    # Generate content
    title = generate_title(pattern)

    # The printed examples are a SAMPLE whose size is a cap, not the key's
    # evidence: `examples_shown: 3` beside `sessions: 111` says "3 of 111", and
    # `examples_shown: 3` beside `sessions: 3` says "all of them". Without the
    # numerator the two are indistinguishable, and consolidation's Phase 1.3
    # persistence test ("2+ dated snapshots — persistent, not a one-off") was
    # comparing copies of one sample across snapshots (#1231, clause 2).
    examples_shown = len(pattern["examples"])

    if pattern["type"] == "error":
        error_rate = pattern["total_calls"] / len(pattern["sessions"]) if pattern["sessions"] else 0
        buckets = pattern.get("merged_buckets") or [_bucket_row(pattern)]
        bucket_rows = "".join(
            f"| `{b['params_signature']}` | {b['occurrences']} | {b['sessions']} |\n"
            for b in buckets
        )
        growth_note = (
            "\nWhile the ledger's stored baseline for this key is a single bucket's count, "
            "the >10x growth reopen does not evaluate against a merged key (`verdict_for`): "
            "the numbers have different denominators. The verdict still binds on its own "
            "terms and the 60-day expiry still applies. Re-keying the coarse rows per "
            "signature is a human decision (#515)."
            if len(buckets) > 1 else ""
        )
        content = f"""---
candidate: true
pattern: {pattern_slug}
type: error
occurrences: {pattern["total_calls"]}
examples_shown: {examples_shown}
sessions: {len(pattern["sessions"])}
first_seen: {pattern["first_seen"]}
last_seen: {pattern["last_seen"]}
error_rate: 1.0
signature_buckets: {len(buckets)}
status: {status_value}{verdict_fm}
---

# Skill Candidate: {title}

## Pattern Summary
This pattern captures repeated {pattern['error_type']} errors when using the `{pattern['tool_name']}` tool. 
These errors occur across {len(pattern["sessions"])} distinct sessions, indicating a systematic issue worth addressing.
{len(buckets)} mined signature bucket(s) share the key `{pattern_slug}`: the two counts above are their sum and their union, and the verdict ledger adjudicates all of them together under that one key (#515).{growth_note}

## Signature Buckets Behind This Key
| Signature | Occurrences | Sessions |
|---|---|---|
{bucket_rows}
## Error Examples
"""
        for i, example in enumerate(pattern["examples"], 1):
            params = example.get("params_summary", {})
            params_str = str(params) if params else "N/A"
            if len(params_str) > 200:
                params_str = params_str[:200] + "..."
            content += f"""### Example {i} (session: {example["session_key"]}, {example["date"]})
- **Tool:** {example["tool"]}
- **Input:** `{params_str}`
- **Error Type:** {example["error_type"]}
- **Failure Class:** {example.get("failure_class") or "uncoded"}

"""
        
        content += f"""## Suggested Mitigation
{generate_mitigation(pattern['tool_name'], pattern['error_type'])}

## Sessions Affected
"""
        for session in sorted(pattern["sessions"]):
            content += f"- {session}\n"
    elif pattern["type"] == "sequence":
        seq_str = pattern["sequence_str"]
        recovery_flag = "true" if pattern["has_error_recovery"] else "false"
        content = f"""---
candidate: true
pattern: {pattern_slug}
type: sequence
ngram_size: {pattern["ngram_size"]}
sessions: {len(pattern["sessions"])}
examples_shown: {examples_shown}
first_seen: {pattern["first_seen"]}
last_seen: {pattern["last_seen"]}
has_error_recovery: {recovery_flag}
status: {status_value}{verdict_fm}
---

# Skill Candidate: {title}

## Pattern Summary
This {pattern["ngram_size"]}-step tool sequence appears across {len(pattern["sessions"])} distinct sessions: `{seq_str}`.
"""
        borrowed = pattern.get("borrowed_sessions", 0)
        if borrowed:
            # The reported count is the key's own; a reader comparing two
            # snapshots of a windowed key must be able to see what the shorter
            # suffix key already billed (#1327).
            content += (f"{borrowed} further session(s) carried this window too and are "
                        f"counted by its shorter suffix key, not here "
                        f"({pattern.get('total_sessions', len(pattern['sessions']) + borrowed)} "
                        "observed in total).\n")
        if pattern["has_error_recovery"]:
            content += ("This pattern includes **error recovery** — in at least one "
                        "observed instance a failed step is followed, inside this "
                        "window, by a re-attempt of the same call or by a call naming "
                        "the same target (#1327).\n")

        content += "\n## Concrete Examples\n"
        for i, example in enumerate(pattern["examples"], 1):
            content += f"""### Example {i} (session: {example["session_key"]}, {example["date"]})
"""
            for step_idx, step in enumerate(example["steps"], 1):
                params = step.get("params_summary", {})
                params_str = str(params) if params else "N/A"
                if len(params_str) > 200:
                    params_str = params_str[:200] + "..."
                status = "ERROR" if step.get("is_error") else "OK"
                result = step.get("result_summary", "N/A")
                if len(result) > 80:
                    result = result[:80] + "..."
                content += f"""- **Step {step_idx}** (`{step["label"]}`): `{step["tool"]}` [{status}]
  - Input: `{params_str}`
  - Result: {result}
"""
            content += "\n"

        content += f"""## Suggested Skill Encoding
This recurring sequence suggests a multi-step procedure that could be encoded as a single skill:
- Combine the {pattern["ngram_size"]} steps into one atomic operation
- Add pre-condition validation before the first step
- {"Include error recovery logic based on the observed retry pattern" if pattern["has_error_recovery"] else "Add error handling between steps to prevent cascading failures"}

## Sessions Affected
"""
        for session in sorted(pattern["sessions"]):
            content += f"- {session}\n"

    else:
        # Success patterns
        content = f"""---
candidate: true
pattern: {pattern_slug}
type: success
occurrences: {pattern["total_calls"]}
examples_shown: {examples_shown}
sessions: {len(pattern["sessions"])}
first_seen: {pattern["first_seen"]}
last_seen: {pattern["last_seen"]}
error_rate: {pattern["error_rate"]:.2f}
status: {status_value}{verdict_fm}
---

# Skill Candidate: {title}

## Pattern Summary
This pattern represents a reusable procedural pattern: using `{pattern['tool_name']}` with consistent parameters across {len(pattern["sessions"])} distinct sessions.
This represents a candidate for skill encoding to improve efficiency and consistency.

## Usage Examples
"""
        for i, example in enumerate(pattern["examples"], 1):
            params = example.get("params_summary", {})
            params_str = str(params) if params else "N/A"
            if len(params_str) > 200:
                params_str = params_str[:200] + "..."
            status = "SUCCESS" if not example.get("is_error") else "ERROR"
            content += f"""### Example {i} (session: {example["session_key"]}, {example["date"]})
- **Tool:** {example["tool"]}
- **Status:** {status}
- **Input:** `{params_str}`
- **Result:** {example.get("result_summary", "N/A")[:50]}

"""

        content += f"""## Suggested Skill Encoding
This pattern should be encoded as a skill with:
- Pre-condition checks for required resources
- Standardized parameter handling
- Error recovery strategies

## Sessions Affected
"""
        for session in sorted(pattern["sessions"]):
            content += f"- {session}\n"

    # Write file
    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


def write_index(candidates: list[str], output_dir: Path) -> None:
    """Write/update the INDEX.md file."""
    index_path = output_dir / "INDEX.md"
    
    content = """---
type: index
scope: skill-candidates
---

# Skill Candidates Index

This index tracks all skill candidates discovered through trajectory mining.
Generated by `mine-trajectories.py`.

## Summary

"""
    
    error_count = sum(1 for c in candidates if "error" in c and "seq-" not in c)
    sequence_count = sum(1 for c in candidates if c.startswith("candidate-seq-"))
    success_count = len(candidates) - error_count - sequence_count

    content += f"""- **Total candidates:** {len(candidates)}
- **Error patterns:** {error_count}
- **Success patterns:** {success_count}
- **Sequence patterns:** {sequence_count}
- **Last updated:** {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

## Candidates

"""
    
    for candidate in sorted(candidates):
        filepath = output_dir / candidate
        if filepath.exists():
            # Try to extract frontmatter
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_lines = []
                    in_frontmatter = False
                    for line in f:
                        first_lines.append(line)
                        if line.strip() == '---' and len(first_lines) > 1:
                            if not in_frontmatter:
                                in_frontmatter = True
                            else:
                                break
                    # Parse frontmatter
                    frontmatter = ''.join(first_lines)
                    pattern_match = re.search(r'pattern:\s*(\S+)', frontmatter)
                    pattern = pattern_match.group(1) if pattern_match else "unknown"
                    sessions_match = re.search(r'sessions:\s*(\d+)', frontmatter)
                    sessions = sessions_match.group(1) if sessions_match else "?"
                    status_match = re.search(r'status:\s*(\S+)', frontmatter)
                    status = status_match.group(1) if status_match else "?"
                    
                    content += f"- [{candidate}]({candidate}) — Pattern: `{pattern}` — Sessions: {sessions} — Status: {status}\n"
            except Exception:
                content += f"- [{candidate}]({candidate}) — (could not parse)\n"
        else:
            content += f"- [{candidate}]({candidate}) — (file missing)\n"
    
    content += f"""

## Generation Commands

```bash
# Show statistics without writing files
python3 ~/lloyd/scripts/mine-trajectories.py --stats

# Generate candidates for last 7 days (worker agent only)
python3 ~/lloyd/scripts/mine-trajectories.py --days 7 --agent worker --threshold 2

# Generate for all agents
python3 ~/lloyd/scripts/mine-trajectories.py --days 7 --agent all --threshold 2

# Custom output directory
python3 ~/lloyd/scripts/mine-trajectories.py --days 7 --output-dir ~/custom/output/
```

"""
    
    index_path.write_text(content, encoding="utf-8")


# ── Statistics ───────────────────────────────────────────────────────────────

def print_stats(trajectories: list[dict],
                class_counts: dict | None = None) -> None:
    """Print summary statistics from trajectory data.

    `class_counts` is the tally `load_trajectories` fills. It is printed because a
    run that dropped 1,373 of 1,397 sessions has to say so (#493): a candidate set
    that shrank without an explanation looks exactly like a pipeline that stopped
    working.
    """
    if not trajectories and not class_counts:
        print("No trajectories found.")
        return

    total_tools = 0
    total_errors = 0
    agent_counts = defaultdict(int)
    error_type_counts = defaultdict(int)
    tool_name_counts = defaultdict(int)
    session_class_counts = defaultdict(int)

    for traj in trajectories:
        total_tools += traj.get("tool_count", 0)
        total_errors += traj.get("error_count", 0)
        agent_counts[traj.get("agent_id", "unknown")] += 1
        session_class_counts[traj.get("session_class") or UNCODED_CLASS] += 1

        for et in traj.get("error_tools", []):
            error_type_counts[et.get("error_type", "unknown")] += 1
        
        for tool in traj.get("tools", []):
            tool_name_counts[tool.get("name", "unknown")] += 1
    
    print("=" * 60)
    print("TRAJECTORY MINING STATS")
    print("=" * 60)
    print(f"  Trajectories loaded:  {len(trajectories)}")
    print(f"  Total tool calls:     {total_tools}")
    print(f"  Total errors:         {total_errors}")
    if total_tools > 0:
        print(f"  Error rate:           {total_errors / total_tools:.1%}")
    print()
    print("By agent:")
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        print(f"  {agent:<20} {count}")
    # The class histogram is the one that matters, not the agent one: every corpus
    # row carries `agent_id: lloyd`, so a single-valued agent histogram is the
    # symptom #493 was filed on. Both sides of the exclusion are printed, because a
    # candidate set that shrank without an explanation looks exactly like a pipeline
    # that stopped working.
    print()
    print("By session class:")
    for cls, count in sorted(session_class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<20} {count}")
    if class_counts and class_counts.get("dropped"):
        print()
        print("By session class dropped by the exclusion:")
        for cls, count in sorted(class_counts["dropped"].items(),
                                 key=lambda x: -x[1]):
            print(f"  {cls:<20} {count}")
    print()
    print("Top tools:")
    for name, count in sorted(tool_name_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:<30} {count}")
    print()
    print("Error types:")
    for etype, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<20} {count}")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

# ── What a zero-trajectory run means (backlog #998) ──────────────────────────
#
# A run that loads nothing used to be indistinguishable from a productive one in
# three ways at once: it printed an ordinary SUMMARY, it exited 0, and it
# rewrote `INDEX.md` to `Total candidates: 0`. The third is the damaging one —
# `write_index` lists only what the current run emitted, so a zero run replaces
# the index a reader or downstream agent consults, and `_pipeline/` is
# gitignored (`.gitignore:25`), so no diff ever shows it. Measured 2026-09-16:
# the live index read `Total candidates: 0` while 3,920 `candidate-*.md` files
# sat in the same directory.
#
# The two empties are different events and must not share an exit status:
#   * no trajectory rows in the window at all — a legitimately quiet day. Exit 0,
#     said out loud in one line.
#   * rows were in the window and the session-class exclusion plus `--agent`
#     selected none of them — a filter/config error. Exit non-zero naming the
#     filter and every count it excluded, and leave the index alone.
# `--agent worker` is the standing instance of the second: #493 reclassified
# `worker` as machine class, so the documented default selects nothing by design.

EMPTY_WINDOW_MESSAGE = ("Window empty: no trajectory rows in the period read; "
                        "nothing to mine, no index written.")
FILTER_SELECTED_NOTHING_EXIT = 2


def empty_corpus_exit_code(trajectories: list[dict], window: dict,
                           class_counts: dict, agent: str,
                           exclude_machine: bool) -> int | None:
    """Decide what a load that selected nothing means, and say it.

    Returns `None` when anything was loaded — the ordinary path. Otherwise
    returns the process exit status: 0 for a window that held no rows at all,
    `FILTER_SELECTED_NOTHING_EXIT` when rows existed and the selectors admitted
    none of them. The caller must not write the index in either zero case, which
    is why this runs before `emit_candidates` rather than after it.

    The two messages are deliberately different strings: a reader grepping the
    nightly log for one must not find the other, because "nothing happened today"
    and "your filter is wrong" want opposite responses.
    """
    if trajectories:
        return None

    rows_seen = window.get("rows", 0)
    if rows_seen == 0:
        print(f"  {EMPTY_WINDOW_MESSAGE}", file=sys.stderr)
        return 0

    class_dropped = class_counts.get("dropped", {})
    agent_dropped = class_counts.get("agent-dropped", {})
    n_class = sum(class_dropped.values())
    n_agent = sum(agent_dropped.values())
    exclusion = "on" if exclude_machine else "off"
    print(f"  ERROR: --agent {agent} selected 0 of {rows_seen} row(s) in the "
          f"window; no candidates written and INDEX.md untouched.",
          file=sys.stderr)
    print(f"    excluded by --agent {agent}: {n_agent} row(s)", file=sys.stderr)
    for aid, count in sorted(agent_dropped.items(), key=lambda x: -x[1]):
        print(f"      agent-excluded {aid}: {count}", file=sys.stderr)
    print(f"    excluded by the session-class exclusion ({exclusion}): "
          f"{n_class} row(s)", file=sys.stderr)
    for cls, count in sorted(class_dropped.items(), key=lambda x: -x[1]):
        print(f"      dropped {cls}: {count}", file=sys.stderr)
    return FILTER_SELECTED_NOTHING_EXIT


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine trajectory JSONL files for skill candidates."
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Process trajectories from the last N days (default: 7)"
    )
    parser.add_argument(
        "--agent", type=str, default="worker",
        help="Filter by agent: worker, main, or all (default: worker)"
    )
    parser.add_argument(
        "--threshold", type=int, default=2,
        help="Minimum distinct sessions for a pattern to qualify (default: 2)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print summary statistics without writing files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: ~/obsidian/skills/candidates/)"
    )
    parser.add_argument(
        "--trajectory-dir", type=str, default=None,
        help="Override the trajectory JSONL directory "
             "(default: ~/lloyd/_pipeline/trajectories)"
    )
    parser.add_argument(
        "--sessions-dir", type=str, default=None,
        help="Override the session store the session-class join reads "
             "(default: ~/lloyd/sessions). Every row's class is resolved through "
             "that store, so this is what decides what the exclusion filters on."
    )
    parser.add_argument(
        "--include-machine", action="store_true",
        help="Mine every session class, including worker/autonomy/smoke/test/"
             "uncoded traffic. Off by default: the gate qualifies on distinct "
             "sessions, so loop cadence otherwise qualifies the loop's own "
             "patterns (#493) and scripted ids qualify as human work (#1143)."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.trajectory_dir:
        set_trajectory_dir(Path(args.trajectory_dir))
    if args.sessions_dir:
        set_session_store_dir(Path(args.sessions_dir))

    # Load trajectories
    print(f"Loading trajectories from last {args.days} days...", file=sys.stderr)
    class_counts: dict = {}
    window: dict = {}
    trajectories = load_trajectories(days=args.days, agent_filter=args.agent,
                                     exclude_machine=not args.include_machine,
                                     class_counts=class_counts,
                                     window=window)
    print(f"  Loaded {len(trajectories)} trajectory(ies)", file=sys.stderr)
    dropped = sum(class_counts.get("dropped", {}).values())
    # The label is what `skills/trajectory-skill-mining/SKILL.md` quotes, so it
    # stays parseable. It reads loosely on purpose — the dropped tally carries
    # `test` and `uncoded` alongside the machine classes, and the per-class lines
    # below are the attributable part.
    print(f"  Machine-class sessions dropped: {dropped}", file=sys.stderr)
    for cls, count in sorted(class_counts.get("dropped", {}).items(),
                             key=lambda x: -x[1]):
        print(f"    dropped {cls}: {count}", file=sys.stderr)

    # Decide before anything is written whether a zero-load run is a quiet day or
    # a filter that selected nothing (#998). Both print; only the first exits 0;
    # neither reaches `write_index`.
    zero_exit = empty_corpus_exit_code(trajectories, window, class_counts,
                                       agent=args.agent,
                                       exclude_machine=not args.include_machine)

    if args.stats:
        print_stats(trajectories, class_counts=class_counts)
        return zero_exit if zero_exit is not None else 0

    if zero_exit is not None:
        return zero_exit
    
    # Mine patterns
    print("Mining error patterns...", file=sys.stderr)
    error_patterns = mine_error_patterns(trajectories, threshold=args.threshold)
    print(f"  Found {len(error_patterns)} qualifying error pattern(s)", file=sys.stderr)
    
    print("Mining success patterns...", file=sys.stderr)
    success_patterns = mine_success_patterns(trajectories, threshold=args.threshold)
    print(f"  Found {len(success_patterns)} qualifying success pattern(s)", file=sys.stderr)

    print("Mining sequence patterns...", file=sys.stderr)
    # Sequences need a higher threshold than individual tools — basic tool
    # combos (read→edit, glob→read) are inherently frequent but boring.
    # Use max(3, threshold) to filter noise while still respecting --threshold.
    seq_threshold = max(3, args.threshold)
    sequence_patterns = mine_sequence_patterns(trajectories, threshold=seq_threshold)
    print(f"  Found {len(sequence_patterns)} qualifying sequence pattern(s)", file=sys.stderr)

    # Write candidates. `emit_candidates` applies `is_emittable`, so the
    # parameter-key-set class is counted as mined but never reaches skill
    # authoring — the runbook used to dispose of it by hand, one
    # `review_reason` at a time.
    all_patterns = error_patterns + success_patterns + sequence_patterns
    # Emission units, not mined buckets: `emit_candidates` merges error patterns
    # onto their candidate key, so this is the list its `written` result is 1:1 with
    # in this order (`write_candidate_file` returns None exactly when `is_emittable`
    # is False). Merging once here and handing the units to the emitter — rather than
    # letting it merge a second copy — is what makes the zip below pair a pattern with
    # its own file by construction instead of by two calls agreeing.
    emitted = merge_error_patterns(all_patterns)
    written = emit_candidates(emitted, output_dir)
    assert len(written) == len([p for p in emitted if is_emittable(p)]), (
        "emit_candidates and its emitted-pattern list disagree, so the per-key "
        "report below would name the wrong files")
    candidate_files = [os.path.basename(p) for p in written]
    # Mined error buckets folded onto their key, and emission units the gate refused
    # — two different reasons for a file not existing, and the old single
    # `len(all_patterns) - len(written)` conflated them the moment the merge existed.
    merged_away = len(all_patterns) - len(emitted)
    # Counted from the gate, not subtracted from `written`. The arithmetic form was
    # right only by coincidence of shape, never by measuring the gate, and a
    # de-duplicated or collided `written` turns a pattern into a phantom suppression
    # (measured 2026-09-15: 1482 patterns, 1234 attempts, 248 reported, 248 refused by
    # `is_emittable`). Counting `is_emittable` over the emission units reports the
    # thing the line claims (#1131 clause 5) in the units the merge emits (#515).
    suppressed = sum(1 for p in emitted if not is_emittable(p))

    # Keys the verdict ledger blocked. Reported, never silent: the number plus one
    # reason line per key is what turns "zero proposed" from an unexplained gap into a
    # decision a reader can check (#530; #391 recorded the silent-zero symptom).
    # One line per KEY, each naming how many mined signature buckets it gates —
    # a coarse verdict suppressing 19 patterns must not read as one judgement
    # about one pattern (#515).
    blocked = superseded_pattern_keys(emitted)
    notes = {candidate_pattern_key(p): merge_note(p) for p in emitted}

    for pattern, filepath in zip((p for p in emitted if is_emittable(p)), written):
        print(f"  Written: {filepath} [{candidate_pattern_key(pattern)}: "
              f"{merge_note(pattern)}]", file=sys.stderr)
    print(f"  Mined error patterns merged onto a shared key: {merged_away}",
          file=sys.stderr)
    print(f"  Suppressed as non-skill candidates: {suppressed}", file=sys.stderr)
    print(f"  Superseded by verdict: {len(blocked)}", file=sys.stderr)
    for key in blocked:
        print(f"    superseded: {key} ({notes.get(key, 'count unknown')})",
              file=sys.stderr)

    # Write index
    write_index(candidate_files, output_dir)
    print(f"  Updated: {output_dir / 'INDEX.md'}", file=sys.stderr)
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Trajectories read:    {len(trajectories)}")
    print(f"  Error patterns:       {len(error_patterns)}")
    print(f"  Success patterns:     {len(success_patterns)}")
    print(f"  Sequence patterns:    {len(sequence_patterns)}")
    print(f"  Candidates written:   {len(candidate_files)}")
    print(f"  Error patterns merged onto a shared key: {merged_away}")
    print(f"  Suppressed (non-skill): {suppressed}")
    print(f"  Superseded by verdict: {len(blocked)}")
    print(f"  Output directory:     {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
