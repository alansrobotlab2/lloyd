#!/usr/bin/env python3
"""
Skill-lint — advisory quality sweep over ~/obsidian/skills/*/SKILL.md.

Writes `~/obsidian/autonomy/skill-lint-report.md` with eight categories:
  1. DEAD              — unparseable frontmatter or desc+tags both empty (never fires)
  2. MISSING_DESC      — has tags but no description (fires via tag/name only)
  3. DRIFT             — description present but output-framed not trigger-framed
  4. DUPLICATE         — near-duplicate skill names (ranking noise)
  5. STALE             — mtime > 90 days and status != active (candidate for removal)
  6. PHANTOM_TOOL      — names a tool the aggregator does not advertise
  7. MISSING_SCRIPT    — cites a repo script that is not in the tree
  8. INJECTION_PATTERN — body instructs acting on remotely hosted instructions or
                        config, or pipes remote content into a shell (#677)

It also counts authorship (#774): how many live skills carry a `written_by:`
front-matter key naming an unattended job, how many say `interactive`, and how
many say nothing. That is a measurement, not a finding, so it never enters the
category table or the Clean verdict.
It also measures size (#624): body lines and characters per live skill
(front matter excluded), library percentiles and a count over `MAX_BODY_LINES`,
plus the before/after size of the sampled spill skills read from the vault's git
history. Like authorship, that is a measurement beside the table, not a finding.

Advisory only. No automatic deletion or rewrites. Exit 0 always (so nightly
pipeline doesn't fail on lint findings).

"Advisory" describes this script, not the class: every category that matters also
has a hard gate in the suite, because a report nobody reads is not a check.
PHANTOM_TOOL → `tests/test_skill_tool_names.py`, MISSING_SCRIPT →
`tests/test_skill_script_existence.py`, INJECTION_PATTERN →
`tests/test_skill_lint_gates.py`.

Origin: Task #334. Methodology documented in that task's description.
"""
from __future__ import annotations

import datetime as dt
import difflib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

try:
    import yaml  # type: ignore
except ImportError:
    print("ERROR: pyyaml not available. Run from .venvs/lloyd.", file=sys.stderr)
    sys.exit(2)


# The one definition of a live skill lives in `agent_mcp.skills` (#1294), and this
# script is what task #70 runs *by path* — `python …/scripts/skill_lint.py`, which
# puts `scripts/` on `sys.path` and not the repo root. Without the insertion the
# import below raises `ModuleNotFoundError: No module named 'agent_mcp'` and the
# lint silently keeps counting its own directory listing, which is the defect this
# line exists to close: 194 linted on 2026-09-20, five of them retired skills.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_mcp.skills import iter_active_skills, skill_roots  # noqa: E402

REPORT_PATH = Path.home() / "obsidian" / "autonomy" / "skill-lint-report.md"

STALE_DAYS = 90
DUPLICATE_EDIT_RATIO_THRESHOLD = 0.85  # difflib ratio; 1.0 is identical

# Trigger-condition markers: presence of any of these in the first sentence
# of the description is evidence the description tells the LLM *when* to use
# the skill, not just *what it does*.
#
# Matched whole-word, case-insensitive.
TRIGGER_WORDS = frozenset({
    "use", "when", "whenever", "trigger", "triggered", "triggers",
    "before", "after", "if", "unless",
    "each", "every",  # "each time X", "every time X"
})

# Description openers that look imperative (start with a short verb). If the
# first token is ≤10 chars AND not in STOP_OPENERS, we treat it as imperative-
# framed (e.g. "Run …", "Review …", "Build …", "Initialize …"). This catches
# the common Anthropic-skill convention without needing a full POS tagger.
STOP_OPENERS = frozenset({
    "the", "a", "an", "this", "these", "that", "those",
    "lloyd", "lloyd's",  # "Lloyd periodic X" / "Lloyd's FastAPI Y"
    "it", "its",
})


# ── Frontmatter parser (mirrors agent_mcp.skills._parse_frontmatter) ─────────

def _extract_first_block(content: str) -> tuple[str, str, str | None]:
    """Return (frontmatter_text, body, yaml_error_or_None) from the first --- block."""
    if not content.startswith("---"):
        return "", content, None
    end = content.find("\n---", 3)
    if end == -1:
        return "", content, "no closing --- delimiter"
    fm_text = content[3:end]
    return fm_text, content[end + 4:].strip(), None


def parse_frontmatter(content: str) -> tuple[dict, str, str | None]:
    """Return (frontmatter, body, yaml_error_msg_or_None).

    Handles dual-block files: tries the first --- block, and if it lacks
    description/tags, tries a second --- block.  This mirrors the common
    pattern where skills have an obsidian metadata block (segment, tags)
    followed by a skill metadata block (description, tags, category).

    Also mirrors the live scorer's parser so lint verdicts match retrieval
    reality.  NOTE: the live scorer currently only reads the first block —
    this is a known gap (see lint report) and should be fixed in skills.py.
    """
    if not content.startswith("---"):
        return {}, content, None

    fm_text, rest, err = _extract_first_block(content)
    if err:
        return {}, rest, err

    try:
        fm = yaml.safe_load(fm_text) or {}
        if not isinstance(fm, dict):
            # Not a mapping — try second block if it has skill-relevant keys
            pass
        else:
            # Has valid YAML.  Check whether it contains skill-relevant keys.
            # If it does (description or tags), use it directly.
            if "description" in fm or "tags" in fm:
                return fm, rest, None
            # First block has YAML but no skill keys (e.g. segment/tags-only).
            # Try to find a second frontmatter block in the remainder.
    except yaml.YAMLError as exc:
        return {}, rest, f"YAML parse error: {exc.__class__.__name__}: {str(exc)[:120]}"

    # ── Try second block ────────────────────────────────────────────────────
    rest_stripped = rest.lstrip("\n")
    if rest_stripped.startswith("---"):
        fm2_text, body2, err2 = _extract_first_block(rest_stripped)
        if err2:
            return {}, rest, err2
        try:
            fm2 = yaml.safe_load(fm2_text) or {}
            if not isinstance(fm2, dict):
                return {}, rest, "second frontmatter block is not a mapping"
            return fm2, body2, None
        except yaml.YAMLError as exc:
            return {}, rest, f"second block YAML error: {str(exc)[:120]}"

    # No second block — use whatever we got (may be empty)
    return {}, rest, None


# ── Checks ────────────────────────────────────────────────────────────────────

def _first_sentence(text: str) -> str:
    """Return the first sentence (split on '. ', '? ', '! ') or the whole text."""
    if not text:
        return ""
    m = re.search(r"[.!?](?:\s|$)", text)
    return text[: m.start()].strip() if m else text.strip()


def _first_token(text: str) -> str:
    m = re.match(r"\s*([A-Za-z][A-Za-z\-']*)", text or "")
    return m.group(1).lower() if m else ""


def check_description_drift(description: str) -> tuple[bool, str]:
    """Return (is_drift, reason). `is_drift=True` means flag it."""
    if not description or not description.strip():
        return True, "description is empty"

    first = _first_sentence(description)
    if not first:
        return True, "description has no first sentence"

    tokens = re.findall(r"\b\w+\b", first.lower())
    token_set = set(tokens)

    # Path 1: contains a trigger word anywhere in the first sentence → good
    if token_set & TRIGGER_WORDS:
        return False, ""

    # Path 2: first token looks imperative (short verb, not in STOP_OPENERS) → good
    opener = _first_token(first)
    if opener and opener not in STOP_OPENERS and len(opener) <= 10:
        # Nothing here tests that the opener IS a verb: any ≤10-character first
        # token outside STOP_OPENERS passes, so "Production pipeline for …" and
        # "Data pipeline: …" clear DRIFT untested. Task #70's calibration
        # (2026-09-10) measured 102 of 189 descriptions passing this way, which
        # is why CATEGORY_TRUST marks the DRIFT count untrustworthy. Tightening
        # or dropping this path is a calibration trade-off (#903, human clause),
        # not something this comment can promise a later check will catch.
        return False, ""

    return True, f"first sentence opens with {opener!r}, no trigger words present"


REQUIRED_FIELDS = ("description", "tags")


def check_dead(fm: dict, yaml_err: str | None) -> tuple[bool, list[str]]:
    """Return (is_dead, reasons).

    A skill is DEAD if the live scorer (`_score_skill` with
    `require_metadata_hit=True`) would score 0 regardless of query.
    That happens when description AND tags are both empty/missing, since
    `name` falls back to the directory name.
    """
    reasons: list[str] = []
    if yaml_err:
        reasons.append(f"frontmatter: {yaml_err}")

    desc = (fm.get("description") or "").strip()
    tags = fm.get("tags") or []
    if not desc:
        reasons.append("description: missing or empty")
    if not tags:
        reasons.append("tags: missing or empty")

    # DEAD if both desc and tags are empty — the only metadata left would be
    # the directory name, which can still fire, so we don't call it fully
    # dead unless the name is also generic. But two empty fields is enough
    # signal to flag.
    is_dead = (not desc) and (not tags)
    return is_dead, reasons


# ── Phantom tool names ────────────────────────────────────────────────────────
#
# A skill naming a tool that does not exist is worse than no skill: the model
# calls it, gets an unknown-tool error, and takes whatever fallback the skill
# documents. `websearch/SKILL.md` told the model to use `web_search` — a name
# Lloyd has never had — and named Bash + curl as the recovery path, which is
# where the curl habit for web lookups came from (2026-09-04).
#
# tests/test_skill_tool_names.py is the hard gate in CI; this is the scheduled
# half, so drift shows up in the weekly report rather than only when someone
# runs pytest.

PHANTOM_TOOLS = frozenset({
    "web_search", "web_fetch", "web_extract",
    "WebSearch", "WebFetch", "HTTPFetch",
    "mcp____http_search", "mcp____http_fetch",
    "mem_get", "mem_write", "mem_search",
    "delegate_task", "execute_code", "sessions_spawn", "skills_get", "skills_list",
    "write_file", "file_write", "file_edit", "file_read", "read_file",
    "vault_get", "run_bash", "search_files", "add_fact",
    "skill_view",
    "file_glob", "file_grep", "tag_search", "tag_explore",
    "pipeline_dispatch", "chat_send", "sessions_send",
    # Retired on 2026-09-23 as duplicates or subsumed. The four #376 verbs
    # (remember, recall, forget, improve) went too but are English words, so a
    # word-boundary match on them would flag ordinary prose.
    "fact_profile", "fact_check", "browser_type",
    "autoresearch_round", "autoresearch_promote", "autoresearch_bench_add",
    "autoresearch_bench_list", "autoresearch_ledger_query",
})

# Skills whose job is to say these names are not real.
PHANTOM_EXEMPT = frozenset({
    "web-search-and-fetch", "nightly-skills-management", "trajectory-skill-mining",
    "nightly-skill-consolidation", "create-hermes-plugin", "autonomy-task-diagnosis",
    "pipeline-dispatch",
})


# `terminal` was the OpenClaw name for Bash. It is also an ordinary English
# word ("run it from the terminal"), so only tool-shaped usage counts:
# a backticked name, or call syntax.
_TERMINAL_AS_TOOL = re.compile(r"`terminal`|\bterminal\s*\(")

_PHANTOM_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, PHANTOM_TOOLS))) + r")\b")


def check_phantom_tools(name: str, content: str) -> list[str]:
    """Tool names mentioned by this skill that the aggregator does not serve."""
    if name in PHANTOM_EXEMPT:
        return []
    found = set(_PHANTOM_RE.findall(content))
    if _TERMINAL_AS_TOOL.search(content):
        found.add("terminal")
    return sorted(found)


# Two skills are duplicates only if they do the same THING. A similar name is
# the cheap signal; on its own it flags whole naming conventions as duplicates
# — `read-validation-handling` vs `grep-validation-handling` were 0.92 similar
# and about different tools. Descriptions are the real evidence, and since
# 2026-09-04 every skill has one, so both must match.
DUPLICATE_DESC_RATIO_THRESHOLD = 0.60


def find_duplicates(skills: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
    """Near-duplicate pairs from a list of (name, description).

    A pair is reported only when the names are close (`DUPLICATE_EDIT_RATIO_THRESHOLD`)
    AND the descriptions are close (`DUPLICATE_DESC_RATIO_THRESHOLD`). The
    reported score is the name ratio, so the report reads as before.

    A skill with no description falls back to name-only, which keeps the check
    working for anything the MISSING_DESC category has yet to catch.
    """
    out: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for i, (a, desc_a) in enumerate(skills):
        for b, desc_b in skills[i + 1:]:
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            if ratio < DUPLICATE_EDIT_RATIO_THRESHOLD:
                continue
            if desc_a and desc_b:
                d_ratio = difflib.SequenceMatcher(None, desc_a.lower(), desc_b.lower()).ratio()
                if d_ratio < DUPLICATE_DESC_RATIO_THRESHOLD:
                    continue
            out.append((*key, round(ratio, 3)))
    return sorted(out, key=lambda t: -t[2])


def check_stale(skill_path: Path, fm: dict) -> tuple[bool, int]:
    """Return (is_stale, age_days). Active-marked skills are exempt."""
    if (fm.get("status") or "").lower() == "active":
        return False, 0
    try:
        mtime = dt.datetime.fromtimestamp(skill_path.stat().st_mtime)
    except OSError:
        return False, 0
    age = (dt.datetime.now() - mtime).days
    return age > STALE_DAYS, age


# ── authorship (#774) ───────────────────────────────────────────────────────
#
# "Which live skills did an unattended job write?" was answerable only by
# matching vault commit subjects, which breaks the first time a job commits with
# a generic message. The writer knows the answer when it writes, so the four
# writer skills stamp `written_by: {job: <task>, date: <UTC>}` and this counts
# the stamps — no git walk. Files written before the stamp existed carry none and
# are counted as unrecorded, never guessed at.
INTERACTIVE_AUTHOR = "interactive"


def written_by_job(fm: dict) -> str:
    """The `written_by` job a skill declares, or "" when it declares none.

    Accepts the mapping the writer skills emit and a bare string, since a hand
    edit will write `written_by: interactive`; anything else reads as unrecorded
    rather than as a job, so a malformed stamp cannot inflate the machine count.
    """
    value = fm.get("written_by")
    if isinstance(value, dict):
        value = value.get("job")
    return value.strip() if isinstance(value, str) else ""


# ── Runner ────────────────────────────────────────────────────────────────────

# ── named repo scripts that no longer exist ─────────────────────────────────
#
# #376's vault half shipped a day without its code half: the skill named
# `scripts/memory/fact-improvement.py` while that file was still in a round, and
# nothing in the suite noticed, because the only check of its kind was
# `tests/test_memory_improvement.py`, written for that one pair. This is the
# generic version, and it is generic in the direction that matters: it reads the
# skill's own text, so a skill cannot cite a script the tree does not have.
#
# Only the anchored form is a claim: `~/lloyd/scripts/foo.py` or
# `$HOME/lloyd/scripts/foo.py`. A skill that says "write `tests/test_login.py`
# for it" is describing work, not citing this checkout — 40-odd skills say that,
# including Anthropic's own shipped ones — so an unanchored path is not checked.
# The cost of that narrower rule is that a skill citing `scripts/foo.py` bare
# slips through; the gain is that the rule can be ON, which the wide version
# could not be. `tests/test_fact_identity_one_action_one_fact.py` pins both directions.
_REPO_CODE_ROOTS = ("agent_mcp/", "app/", "workers/", "eval/", "scripts/",
                    "tests/", "agent-services/", "web/")
_SCRIPT_PATH_RE = re.compile(
    r"(?:~/?lloyd/|\$HOME/lloyd/)"
    r"((?:agent_mcp|app|workers|eval|scripts|tests|agent-services|web)/"
    r"[\w.\-/]*\.(?:py|sh|js|ts|mjs))(?![\w.\-/])")
_TEMPLATE_PATHS = re.compile(r"[<>{}*]|\.\.\.|path/to|exact/path|example|placeholder",
                             re.I)

# Documented-but-stale references, each one a real drift the rule found on the
# day it was written. Listed so the rule can be enforced from the day it lands
# without the whole suite turning red over findings that predate it — and an
# entry here is a debt with a name, not a blind spot: a NEW absent path fails.
# It is also an entry with a route OUT: #1417 retired
# `tests/test_system_health_check_frontend_endpoint.py` once the
# `system-health-check` skill's only citation of it lost its `~/lloyd/` anchor
# (the 2026-09-23 skill edit), which left an allowance for a path no skill
# cites — the state
# `test_the_absent_script_ledger_only_carries_drift_still_cited` exists to
# catch. The exact set is pinned by
# `test_the_absent_script_ledger_holds_exactly_the_cited_debts`; the
# never-landed test that entry recorded is still owed debt and is named on
# backlog #1417, not here.
KNOWN_ABSENT_SCRIPTS: dict[str, str] = {
    "scripts/memory/extract-session-log.py":
        "historical-knowledge-refresh; superseded by extract-transcript.py",
    "scripts/memory/next-gen-memory/context_bundle.py":
        "memory-path-scoping; directory removed with the next-gen-memory scripts",
    "tests/test_health_skill_docs_live_fleet.py":
        "system-health-check; test never landed",
}


def check_script_paths(content: str, skill_dir: Path | None = None,
                       repo_root: Path | None = None) -> list[dict]:
    """Repo script paths the skill cites that are not in the tree."""
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    out: list[dict] = []
    for match in set(_SCRIPT_PATH_RE.findall(content)):
        if _TEMPLATE_PATHS.search(match):
            continue
        if match.startswith(_REPO_CODE_ROOTS):
            target = repo_root / match
            found = target.exists()
        else:
            found = bool(skill_dir and (skill_dir / match).exists())
        if not found:
            out.append({"path": match,
                        "known_stale": KNOWN_ABSENT_SCRIPTS.get(match, "")})
    return sorted(out, key=lambda d: d["path"])


# ── injection-shaped instructions in the body (#677) ─────────────────────────
#
# Snyk's AI Engineer talk (Manoj Nair, "Through the AI Fog: The Architectural
# Decision Agentic Security Depends On") demoed a shared "competitive analysis"
# skill whose worst finding was not code at all: one line told the agent to load
# its monitoring targets and classification rules from a YAML file hosted on the
# internet, so "even if my skill file doesn't change at all, that is a chance for
# an exploit to occur" — the behaviour changes under someone else's control while
# the skill stays byte-identical. His prescription was a deterministic hook
# rather than a prompt line, because the same vulnerability was found in ~50% of
# frontier-model runs (0.40 F1) while a regex pass runs every time. That is the
# generator-and-validator-can't-be-the-same-system point applied to Lloyd's own
# instruction layer: the nightly consolidator that WRITES skills must not be the
# only thing that CHECKS them, which is why the hard gate lives in
# `tests/test_skill_lint_gates.py` and not in a consolidator prompt.
#
# Two rules, and only two, because a regex is worth exactly its measured
# precision over THIS corpus. Measured on the 187 live skills at landing:
#
#   remote_instruction_fetch  0 matches
#   piped_remote_execution    1 match (`huggingface-hub`, listed below)
#
# The zero is the finding, and the item's own risk clause demands it be reported
# rather than grown: "if step 3 returns zero true positives, close the item with
# that number as the finding rather than stretching the rules until something
# appears." The two rules this item also proposed were measured and dropped:
# secret-echo fired on 19 lines across 14 skills (`code-review`'s
# "Authentication/authorization checks present" is the archetype) and
# writes-outside-declared-roots on 13 lines across 7 (`nightly-reflection-knowledge-write`
# legitimately writes `~/lloyd/_pipeline/**`), both near-zero true positives. A
# gate that is red on day one gets switched off by the next person who sees it,
# so those two are recorded on the item as out of scope instead of shipped as noise.
#
# What this is NOT: a detector. A regex over natural language misses paraphrase
# entirely, so this is a coverage floor and a review queue. Whether Lloyd actually
# obeys injected content is measured at runtime by #590; #547 is an
# activation-time authorizing-context check, #624 a body-length cap, #543 a
# cross-corpus artifact sweep. Same neighbourhood, different questions — do not
# collapse them in a later triage pass.

INJECTION_CATEGORY = "INJECTION_PATTERN"

# A URL whose last path component is a config/instruction extension. The
# extension requirement is load-bearing: the naive "fetch verb + any http(s) URL"
# fired 31 lines across 17 skills, every one of them a benign content fetch
# (`skills/arxiv/SKILL.md:39`'s `http_fetch(url="https://arxiv.org/abs/…")` is the
# archetype), and a gate red on day one is a gate that gets deleted.
_INSTR_EXT = r"\.(?:yaml|yml|json|jsonl|md|txt|toml|conf)(?![\w-])"
_URL = r"https?://[^\s'\"`<>()\[\]|,;]+"
_INSTR_URL = _URL + _INSTR_EXT
# "act on it" verbs, finite forms only. `http_fetch` is deliberately unmatched:
# `_` is a word character, so `\bfetch\b` cannot match inside it — that single
# detail is what keeps every `http_fetch(...)` example in the corpus clean.
_INSTR_VERB = (r"\b(?:fetch|fetches|pull|pulls|load|loads|follow|follows|"
               r"apply|applies|obey|obeys)\b")
# Within one sentence: stop at ". " rather than running to end of line, so a
# sentence that merely links a config file is not also a sentence that says to
# obey it. Verified both directions by tests/test_skill_lint_injection_rules.py.
_WITHIN_SENTENCE = r"(?:(?!\.\s)[^\n])*?"

_PIPE_INTO_SHELL = (r"\b(?:curl|wget)\b[^|\n]*\|\s*"
                    r"(?:sudo\s+)?(?:ba|z|da|k)?sh\b")

#: The rule table. One entry per rule, matched against ONE LINE of a SKILL.md at
#: a time; deleting a key switches that rule off and nothing else, so ablating a
#: rule is a one-line change rather than an edit to the checker. A rule with no
#: measured corpus yield does not belong here (item #677's risk clause), and the
#: exact set is pinned by `test_the_rule_table_is_named_and_each_entry_is_a_compiled_rule`.
INJECTION_RULES: dict[str, re.Pattern] = {
    # Snyk's "especially problematic" finding: fetch instructions/config from a
    # remote URL and act on them.
    "remote_instruction_fetch": re.compile(
        rf"(?:{_INSTR_VERB}{_WITHIN_SENTENCE}{_INSTR_URL}"
        rf"|{_INSTR_URL}{_WITHIN_SENTENCE}{_INSTR_VERB})",
        re.I),
    # Remote content executed without being looked at first. Denoised against the
    # runtime, which already refuses this at `app/harness/safety.py:88-95`: a
    # skill that instructs it is a skill whose documented first step is denied,
    # so the finding is a broken instruction as often as a hostile one.
    "piped_remote_execution": re.compile(_PIPE_INTO_SHELL, re.I),
}

#: Skill name -> the written reason its matched line stays. Not a frozenset:
#: `PHANTOM_EXEMPT` above is one, carries no reason at all, and `KNOWN_ABSENT_SCRIPTS`
#: is the lesson from a name-only ledger — an entry that outlives its citation is a
#: permanently open permit granted by a line nobody can point at in the corpus any
#: more. So every entry here needs a reason AND has to still match, both asserted
#: in `tests/test_skill_lint_gates.py`.
INJECTION_ALLOWLIST: dict[str, str] = {
    "huggingface-hub":
        "Documented one-time installer for the HF CLI, "
        "`curl -LsSf https://hf.co/cli/install.sh | bash -s`, from the vendor's "
        "own domain: listed rather than rewritten so the skill still tells the "
        "reader how to install the tool it is about. The same pipe is denied at "
        "runtime by app/harness/safety.py, so the line is guidance, not a step "
        "the agent can execute.",
}


def check_injection_patterns(skill_name: str, content: str) -> list[dict]:
    """Every injection-shaped line in this SKILL.md, one dict per (line, rule).

    Reported whether or not the skill is allow-listed: a permit hides a finding
    from the gate, never from the report, which is how an entry can be audited
    before it silently outlives the line it was written for. `allow_reason` is
    that audit trail, carried on the hit itself rather than looked up by a
    renderer that might forget to.
    """
    reason = str(INJECTION_ALLOWLIST.get(skill_name, "") or "")
    out: list[dict] = []
    for line_no, line in enumerate(content.splitlines(), 1):
        for rule_name, rule in INJECTION_RULES.items():
            if rule.search(line):
                out.append({"rule": rule_name, "line_no": line_no, "line": line,
                            "allow_reason": reason})
    return out


def unlisted_injection_findings(result: dict) -> list[dict]:
    """Flat list of injection hits with no allow-list reason: the gate's failure set.

    One function so the CI gate and any future report agree on what "unlisted"
    means, instead of each writing its own filter and drifting.
    """
    return [{"skill": finding["name"], "path": finding["path"], **hit}
            for finding in result.get("injection", [])
            for hit in finding["hits"]
            if not str(hit.get("allow_reason", "")).strip()]


# ── size (#624) ─────────────────────────────────────────────────────────────
#
# "Your skill is really a folder": the SKILL.md body is an index and the detail
# lives in sibling files read on demand. Whether that pays depends on the route.
# The chat injector cuts at `CHAT_SKILL_CUT` chars, so a long body there costs
# nothing extra — it loses its tail instead, which is why `past_chat_cut` is
# reported. The autonomy task prompt and the worker prompt splice the whole file
# in uncapped, so there the size is the cost (`app/skill_embed.py` records which
# route paid what per run). Advisory: the 100-line figure is one team's
# heuristic, and the ceiling is to be set from the measured curve, by a person.

#: The body-length ceiling the SIZE bucket counts against (front matter excluded).
MAX_BODY_LINES = 100

#: `prefetch.SKILL_BODY_MAX`, restated because this script runs by path and
#: importing prefetch pulls in the retrieval stack; a test pins the two equal.
CHAT_SKILL_CUT = 6000

#: The five oversized skills #624 samples for a spill pass. The pass itself is a
#: vault edit; `spill_delta` reports what it has done to their embedded size.
SPILL_SAMPLE = ("powerpoint", "deep-research", "nightly-reflection-knowledge-write",
                "system-health-check", "entity-resolution-sweep")

#: The vault state the spill delta is measured from: the last vault commit on or
#: before this instant, i.e. before any #624 spill landed.
SPILL_BASELINE_BEFORE = "2026-09-24T23:59:59"

_HEADING_RE = re.compile(r"^#{2,3}\s+\S")


def _largest_block(body_lines: list[str]) -> dict:
    """The largest `##`/`###`-delimited block: the first spill candidate."""
    best = {"heading": "", "lines": 0}
    start, heading = 0, "(before first heading)"
    for i, line in enumerate(body_lines + ["## (end)"]):
        if _HEADING_RE.match(line):
            if i - start > best["lines"]:
                best = {"heading": heading, "lines": i - start}
            start, heading = i, line.strip()
    return best


def skill_size(name: str, path: Path, content: str, body: str) -> dict:
    """One skill's SIZE row. `body` is the text after front matter."""
    body = body.strip("\n")
    lines = body.splitlines()
    return {
        "name": name,
        "path": str(path),
        "body_lines": len(lines),
        "body_chars": len(body),
        # What the autonomy and worker prompts embed: the file, front matter and all.
        "file_chars": len(content),
        "over_cap": len(lines) > MAX_BODY_LINES,
        "past_chat_cut": max(0, len(content) - CHAT_SKILL_CUT),
        "largest_block": _largest_block(lines),
    }


def _percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))  # ceil(n * q / 100)
    return ordered[int(rank) - 1]


def size_summary(rows: list[dict]) -> dict:
    """The SIZE bucket: library percentiles, the over-cap count, every row."""
    lines = [r["body_lines"] for r in rows]
    chars = [r["body_chars"] for r in rows]
    return {
        "max_body_lines": MAX_BODY_LINES,
        "chat_skill_cut": CHAT_SKILL_CUT,
        "count": len(rows),
        "over_cap": sum(1 for r in rows if r["over_cap"]),
        "p50_lines": _percentile(lines, 50),
        "p90_lines": _percentile(lines, 90),
        "max_lines": max(lines, default=0),
        "p50_chars": _percentile(chars, 50),
        "p90_chars": _percentile(chars, 90),
        "max_chars": max(chars, default=0),
        "skills": sorted(rows, key=lambda r: (-r["body_lines"], r["name"])),
    }


def _git_show(repo: Path, rev: str, rel: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{rel}"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def spill_delta(vault: Path, names: Sequence[str] = SPILL_SAMPLE,
                before: str = SPILL_BASELINE_BEFORE) -> dict:
    """Before/after size of the sampled skills as the autonomy prompt embeds them.

    "Before" is `skills/<name>/SKILL.md` at the last vault commit on or before
    `before`; "after" is the file on disk now. Read-only git. The delta is the
    saving #624 can actually claim — on the uncapped routes; on the capped chat
    route a shorter body changes what survives the cut, not what it costs. A
    delta of 0 means the spill has not been done, and the report says so.
    """
    base = None
    try:
        out = subprocess.run(["git", "-C", str(vault), "rev-list", "-1",
                              f"--before={before}", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            base = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        base = None
    rows = []
    for name in names:
        rel = f"skills/{name}/SKILL.md"
        then = _git_show(vault, base, rel) if base else None
        path = vault / rel
        try:
            now = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            now = None
        now_body = parse_frontmatter(now)[1].strip("\n") if now is not None else None
        then_body = parse_frontmatter(then)[1].strip("\n") if then is not None else None
        rows.append({
            "name": name,
            "before_chars": len(then) if then is not None else None,
            "after_chars": len(now) if now is not None else None,
            "delta_chars": (len(now) - len(then)) if now is not None and then is not None else None,
            "before_body_lines": len(then_body.splitlines()) if then_body is not None else None,
            "after_body_lines": len(now_body.splitlines()) if now_body is not None else None,
            "siblings": sorted(p.name for p in path.parent.glob("*.md")
                               if p.name != "SKILL.md") if path.parent.is_dir() else [],
        })
    measured = [r["delta_chars"] for r in rows if r["delta_chars"] is not None]
    return {"baseline_rev": base, "baseline_before": before, "skills": rows,
            "total_delta_chars": sum(measured) if measured else None}


def lint(skill_records: Optional[Sequence] = None) -> dict:
    """Lint every *live* skill — the set `agent_mcp.skills.iter_active_skills` owns.

    This used to be a sixth walk, over a `SKILLS_DIR` it hardcoded and with no
    quarantine rule at all: it linted retired-in-place skills as if they were live,
    so the weekly report's `total` (194 on 2026-09-20) was computed over a set no
    loader ever uses, and it never saw the second configured root.

    Narrowing the set is only admissible because every finding below is computed
    from the same walked records: DEAD, MISSING_DESC, DRIFT, DUPLICATE, STALE,
    PHANTOM_TOOL, MISSING_SCRIPT and INJECTION_PATTERN all still fire for a live
    skill that has them.
    What stops being reported is a defect in a skill the model can no longer reach —
    a retired skill cannot mislead anyone, and its findings would be permanently
    unactionable noise in a report a human reads.

    `skill_records` takes records in `iter_active_skills`' own shape, so a test can
    point the whole loop at a temp skills root and prove the injection gate can
    fail (#677 clause 4) without this function growing a second, private corpus
    walk — the sixth-walk bug the paragraph above is about. The live nightly path
    passes nothing and is unchanged.
    """
    if skill_records is None:
        active = list(iter_active_skills())
        roots_walked = skill_roots()
    else:
        # The `Scanned N live skills in <roots>` line is a measurement, so when a
        # caller brings its own records the roots printed are where those records
        # came from, not the configured pair this process would have walked.
        active = list(skill_records)
        roots_walked = sorted({r.directory.parent for r in active})

    dead: list[dict] = []
    missing_desc: list[dict] = []
    drift: list[dict] = []
    stale: list[dict] = []
    phantom: list[dict] = []
    missing_script: list[dict] = []
    injection: list[dict] = []
    authors: Counter = Counter()
    unrecorded: list[str] = []
    sizes: list[dict] = []
    skills: list[tuple[str, str]] = []
    total = 0

    for record in active:
        entry = record.directory
        skill_file = record.skill_file
        total += 1
        # description filled in below once frontmatter is parsed

        try:
            content = skill_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            dead.append({
                "name": entry.name,
                "path": str(skill_file),
                "reasons": [f"unreadable: {exc}"],
            })
            continue

        fm, body, yaml_err = parse_frontmatter(content)

        # Counted before the dead check: a dead skill is still in the library,
        # and who wrote it is the first question when it has to be fixed.
        job = written_by_job(fm)
        if job:
            authors[job] += 1
        else:
            unrecorded.append(entry.name)

        is_dead, dead_reasons = check_dead(fm, yaml_err)
        if is_dead:
            dead.append({
                "name": entry.name,
                "path": str(skill_file),
                "reasons": dead_reasons,
            })
            continue  # drift check is redundant when already dead

        desc = (fm.get("description") or "").strip()
        skills.append((entry.name, desc))
        sizes.append(skill_size(entry.name, skill_file, content, body))
        tags = fm.get("tags") or []
        if not desc and tags:
            # Has tags but no description → MISSING_DESC (fires via tag/name only,
            # loses the 2× description weight in the scorer)
            missing_desc.append({
                "name": entry.name,
                "path": str(skill_file),
                "tags": tags,
            })
        elif desc:
            is_drift, drift_reason = check_description_drift(desc)
            if is_drift:
                drift.append({
                    "name": entry.name,
                    "path": str(skill_file),
                    "description": desc[:200] + ("…" if len(desc) > 200 else ""),
                    "reason": drift_reason,
                })

        bad_tools = check_phantom_tools(entry.name, content)
        if bad_tools:
            phantom.append({
                "name": entry.name,
                "path": str(skill_file),
                "tools": bad_tools,
            })

        # Allow-listed hits are kept in the payload and the report: the permit is
        # on the gate, not on being seen. `unlisted_injection_findings` is what
        # `tests/test_skill_lint_gates.py` fails on.
        inj_hits = check_injection_patterns(entry.name, content)
        if inj_hits:
            injection.append({
                "name": entry.name,
                "path": str(skill_file),
                "hits": inj_hits,
            })

        bad_scripts = check_script_paths(content, skill_dir=entry)
        live_scripts = [b for b in bad_scripts if not b["known_stale"]]
        if live_scripts:
            missing_script.append({
                "name": entry.name,
                "path": str(skill_file),
                "scripts": live_scripts,
            })

        is_stale, age = check_stale(skill_file, fm)
        if is_stale:
            stale.append({
                "name": entry.name,
                "path": str(skill_file),
                "age_days": age,
                "status": fm.get("status", "(unset)"),
            })

    duplicates = find_duplicates(skills)

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "total": total,
        # Which roots `total` was counted over. The report used to print a hardcoded
        # `~/obsidian/skills/`, which was a claim rather than a measurement now that
        # the walker follows `config.yaml skills.directories` and may scan two roots.
        "roots": [str(r) for r in roots_walked],
        # #624. Over the skills that parse (a DEAD skill has no body boundary to
        # measure from); a measurement, so it never enters the verdict.
        "size": size_summary(sizes),
        "dead": dead,
        "missing_desc": missing_desc,
        "drift": drift,
        "duplicates": duplicates,
        "stale": stale,
        "phantom": phantom,
        "missing_script": missing_script,
        "injection": injection,
        "authorship": {
            "by_job": dict(sorted(authors.items())),
            "machine_written": sum(n for j, n in authors.items()
                                   if j != INTERACTIVE_AUTHOR),
            "interactive": authors.get(INTERACTIVE_AUTHOR, 0),
            "unrecorded": sorted(unrecorded),
        },
    }


# ── Report writer ─────────────────────────────────────────────────────────────

#: Per-category answer to "is a 0 in this row trustworthy?", rendered as the
#: report table's fourth column and repeated in the verdict when nothing fired.
#:
#: Task #70's 2026-09-10 calibration instrumented every check and asked, per
#: category, *could this have flagged anything today?* Two could not: DRIFT
#: accepts 53% of the library through path 2 above without testing it, and
#: STALE returns before its age comparison for every skill (194/194 carry
#: `status: active`; no mtime in the library is older than a month either).
#: The qualified table was hand-written into the committed report (`218882b5`)
#: and lost on the next run, because `main` overwrites the file wholesale — so
#: the answer lives here, where the regeneration cannot drop it (#903).
#:
#: Keyed on the category name as the table prints it. Every row rendered by
#: `render_report` must have an entry, and every entry must be a row:
#: `tests/test_skill_lint_report_trust.py` asserts both, so a new category
#: cannot ship with an unqualified zero. `verdict` is one of TRUST_VERDICTS;
#: only `"no"` makes the category untrustworthy for the verdict below.
TRUST_VERDICTS = ("yes", "mostly", "no")
CATEGORY_TRUST: dict[str, tuple[str, str]] = {
    "DEAD": ("yes", "verified by independent recount over every scanned file"),
    "MISSING_DESC": ("yes", "same recount; every live skill parses and carries a description"),
    "DRIFT": ("no", "descriptions whose first token is ≤10 characters pass a "
                    "length heuristic, verb or not, and are never tested — "
                    "0 means 'the rest contain a trigger word', not 'none drift'"),
    "DUPLICATE": ("mostly", "the name + description double gate suppresses a real "
                            "pair (`periodic-memory-capture-dee`/`-lloyd`), which is "
                            "a judgment call the report cannot make"),
    "STALE": ("no", "every skill is marked `status: active` and `check_stale` "
                    "returns before the age check, so no input reaches the "
                    "comparison — 0 carries no information"),
    "PHANTOM_TOOL": ("yes", "positive controls fire and the count suppresses the "
                            "Clean verdict; the PHANTOM_EXEMPT skills are never scanned"),
    "MISSING_SCRIPT": ("yes", "the test suite asserts the corpus-wide count, so an "
                              "absent path has already turned the tests red"),
    INJECTION_CATEGORY: ("yes", "every line of every skill is matched, but a match is a "
                                "shape, not a risk — a regex misses paraphrase, so 0 "
                                "means 'no match', never 'no risk'"),
}

TRUST_MARK = {"yes": "✅ yes", "mostly": "⚠️ mostly", "no": "❌ no"}


def untrustworthy_categories() -> list[str]:
    """Categories whose 0 says nothing, in table order."""
    return [c for c, (verdict, _) in CATEGORY_TRUST.items() if verdict == "no"]


def trust_cell(category: str) -> str:
    verdict, cause = CATEGORY_TRUST[category]
    return f"{TRUST_MARK[verdict]} — {cause}"


def render_size(result: dict) -> list[str]:
    """The SIZE section (#624): summary, the sampled spill delta, every skill.

    Beside the category table, never a row in it: over half the library is over
    the cap, so as a finding it would read the same every night and hide the
    categories that can fail. Every skill gets a row so the per-skill number is
    in the report, not only the JSON.
    """
    size = result.get("size")
    if not size:
        return []
    out = [f"### SIZE — body length (advisory; cap {size['max_body_lines']} lines)", ""]
    out.append(f"Over the cap: **{size['over_cap']}** of {size['count']} · "
               f"body lines p50 **{size['p50_lines']}**, p90 **{size['p90_lines']}**, "
               f"max **{size['max_lines']}** · body chars p50 {size['p50_chars']}, "
               f"p90 {size['p90_chars']}, max {size['max_chars']}.")
    out.append("")
    out.append(f"The chat injector keeps the first {size['chat_skill_cut']} chars of a "
               "skill, so there a long body loses its tail rather than costing more; "
               "the autonomy and worker prompts embed the whole file for the run.")
    out.append("")
    delta = size.get("spill_delta")
    if delta:
        total = delta.get("total_delta_chars")
        out.append(f"Sampled spill (#624), embedded chars vs vault `{(delta.get('baseline_rev') or 'no baseline')[:10]}` "
                   f"(last commit before {delta.get('baseline_before')}): "
                   f"**{total if total is not None else 'not measured'}**"
                   + (" — the spill has not been done yet." if total == 0 else "."))
        out.append("")
        out.append("| skill | chars before | chars now | delta | body lines before → now | sibling files |")
        out.append("|---|---|---|---|---|---|")
        for r in delta.get("skills", []):
            sib = ", ".join(f"`{n}`" for n in r.get("siblings") or []) or "—"
            out.append(f"| `{r['name']}` | {r['before_chars']} | {r['after_chars']} | "
                       f"{r['delta_chars']} | {r['before_body_lines']} → {r['after_body_lines']} | {sib} |")
        out.append("")
    out.append("| skill | body lines | body chars | over cap | chars past chat cut | largest block |")
    out.append("|---|---|---|---|---|---|")
    for r in size.get("skills", []):
        block = r.get("largest_block") or {}
        heading = str(block.get("heading", "")).replace("|", "\\|")
        out.append(f"| `{r['name']}` | {r['body_lines']} | {r['body_chars']} | "
                   f"{'yes' if r['over_cap'] else ''} | {r['past_chat_cut'] or ''} | "
                   f"{heading} ({block.get('lines', 0)} lines) |")
    out.append("")
    return out


def render_report(result: dict) -> str:
    lines: list[str] = []
    ts = result["generated_at"]
    total = result["total"]
    n_dead = len(result["dead"])
    n_missing = len(result.get("missing_desc", []))
    n_drift = len(result["drift"])
    n_dup = len(result["duplicates"])
    n_stale = len(result["stale"])
    n_phantom = len(result.get("phantom", []))
    n_scripts = len(result.get("missing_script", []))
    injection = result.get("injection", [])
    n_inj = len(injection)
    # Per-rule counts, computed from the findings rather than stored beside them,
    # so a rule that ran and found nothing and a rule that never ran both read as
    # 0 and neither can be silently dropped from the table below.
    inj_lines = Counter(hit["rule"]
                        for finding in injection for hit in finding["hits"])
    inj_skills: Counter = Counter()
    for finding in injection:
        for rule_name in {hit["rule"] for hit in finding["hits"]}:
            inj_skills[rule_name] += 1

    lines.append(f"# Skill Lint Report — {ts}")
    lines.append("")
    roots = result.get("roots") or []
    where = ", ".join(f"`{r}`" for r in roots) if roots else "`~/obsidian/skills/`"
    lines.append(f"Scanned **{total}** live skills in {where}.")
    lines.append("")
    # (category, gloss, count, action). The fourth column is looked up by the
    # category name, so a row added here without a CATEGORY_TRUST entry raises
    # KeyError on the first render rather than printing an unqualified zero.
    rows = [
        ("DEAD", "never fires", n_dead, "fix frontmatter or remove"),
        ("MISSING_DESC", "tags only, no description", n_missing, "add trigger-condition description"),
        ("DRIFT", "output-framed description", n_drift, "rewrite first sentence in trigger-condition form"),
        ("DUPLICATE", "near-duplicate names", n_dup, "resolve ownership, merge, or rename"),
        ("STALE", f">{STALE_DAYS}d mtime, status ≠ active", n_stale, "review for removal"),
        ("PHANTOM_TOOL", "names a tool that does not exist", n_phantom, "replace with the real tool name"),
        ("MISSING_SCRIPT", "names a repo script absent from the tree", n_scripts, "land the script or drop the citation"),
        (INJECTION_CATEGORY, "instructs acting on remote instructions/config, or pipes remote content into a shell",
         n_inj, "rewrite to name a local/pinned step, or list in INJECTION_ALLOWLIST with a reason"),
    ]
    lines.append("| category | count | action | is this count trustworthy? |")
    lines.append("|---|---|---|---|")
    for category, gloss, count, action in rows:
        lines.append(f"| {category} ({gloss}) | **{count}** | {action} | {trust_cell(category)} |")
    lines.append("")
    # One row per rule in the table, always, including the rules that matched
    # nothing. `n_phantom` is the precedent for why: until 2026-09-11 the count
    # printed while the section did not, and a reader could not tell "no findings"
    # from "the check is not wired up". #677's remote-instruction rule matches zero
    # live skills today, so without these rows its zero would be exactly that
    # ambiguity — and the whole point of shipping it is that the number gets
    # re-measured weekly instead of being asserted once at implementation time.
    lines.extend(render_size(result))
    lines.append(f"### {INJECTION_CATEGORY} — per-rule counts")
    lines.append("")
    lines.append("| rule | matched lines | skills |")
    lines.append("|---|---|---|")
    for rule_name in sorted(INJECTION_RULES):
        lines.append(f"| `{rule_name}` | {inj_lines.get(rule_name, 0)} "
                     f"| {inj_skills.get(rule_name, 0)} |")
    lines.append("")
    # A measurement beside the table, never a row in it: 0 of 194 skills carried
    # `written_by` when the key was introduced, so as a finding it would have
    # read 194 on every run and hidden the categories that can fail.
    authorship = result.get("authorship") or {}
    n_unrecorded = len(authorship.get("unrecorded", []))
    lines.append("### Authorship (`written_by`)")
    lines.append("")
    lines.append(f"Machine-written live skills: **{authorship.get('machine_written', 0)}** · "
                 f"interactive: **{authorship.get('interactive', 0)}** · "
                 f"no `written_by`: **{n_unrecorded}** of {total}.")
    by_job = authorship.get("by_job") or {}
    if by_job:
        lines.append("")
        lines.append("| job | skills |")
        lines.append("|---|---|")
        for job, n in by_job.items():
            lines.append(f"| `{job}` | {n} |")
    lines.append("")
    lines.append("This report is **advisory**. No automatic changes.")
    lines.append("")

    if result.get("error"):
        lines.append(f"⚠️ Error: {result['error']}")
        return "\n".join(lines) + "\n"

    # DEAD
    if n_dead:
        lines.append(f"## DEAD — {n_dead} skills that never score > 0")
        lines.append("")
        lines.append("These skills have missing or unparseable metadata. The live scorer")
        lines.append("(`agent_mcp.skills._score_skill` with `require_metadata_hit=True`)")
        lines.append("returns 0.0 for any query, so they can never be auto-injected.")
        lines.append("")
        for item in result["dead"]:
            lines.append(f"- **{item['name']}**")
            for r in item["reasons"]:
                lines.append(f"  - {r}")
            lines.append(f"  - path: `{item['path']}`")
        lines.append("")

    # MISSING_DESC
    if n_missing:
        lines.append(f"## MISSING_DESC — {n_missing} skills with tags but no description")
        lines.append("")
        lines.append("Description field is missing or empty. The live scorer still matches via")
        lines.append("name and tags, but loses the 2× weight on description hits. In practice")
        lines.append("these fire only when the user's query directly mentions the skill name")
        lines.append("or an exact tag — which is most of the time not what the skill is for.")
        lines.append("")
        lines.append("| name | tags |")
        lines.append("|---|---|")
        for item in result["missing_desc"]:
            tag_str = ", ".join(f"`{t}`" for t in item["tags"][:5])
            if len(item["tags"]) > 5:
                tag_str += f" (+{len(item['tags']) - 5} more)"
            lines.append(f"| `{item['name']}` | {tag_str} |")
        lines.append("")

    # PHANTOM_TOOL
    if n_phantom:
        lines.append(f"## PHANTOM_TOOL — {n_phantom} skills naming tools that do not exist")
        lines.append("")
        lines.append("These skills instruct the model to call a tool the aggregator does not")
        lines.append("advertise. The call fails with an unknown-tool error and the model takes")
        lines.append("whatever fallback the skill documents — which is how the archived")
        lines.append("`websearch` skill (`web_search`, `web_fetch`) taught Lloyd to shell out to")
        lines.append("`curl` for every web lookup. Replace with the real name:")
        lines.append("`http_search` / `http_fetch` / `http_request` for the web, `Read` /")
        lines.append("`Write` / `Edit` / `Grep` / `Glob` for files, `skills_read` for skills,")
        lines.append("`Task` for subagents, `memory_read` / `memory_add` for memory.")
        lines.append("")
        lines.append("| name | phantom tools |")
        lines.append("|---|---|")
        for item in result["phantom"]:
            lines.append(f"| `{item['name']}` | {', '.join(f'`{t}`' for t in item['tools'])} |")
        lines.append("")

    # DRIFT
    if n_drift:
        lines.append(f"## DRIFT — {n_drift} skills with output-form descriptions")
        lines.append("")
        lines.append("First sentence of the description neither starts with a short imperative")
        lines.append(f"verb nor contains a trigger word ({', '.join(sorted(TRIGGER_WORDS))}).")
        lines.append("These still fire but fire less often than they should — the description")
        lines.append("describes *what the skill is* rather than *when to use it*.")
        lines.append("")
        lines.append("**Good form:** `Use this skill to configure X when Y.`")
        lines.append("**Bad form:** `X subsystem that does Y.`")
        lines.append("")
        for item in result["drift"]:
            lines.append(f"- **{item['name']}** — {item['reason']}")
            lines.append(f"  - current: `{item['description']}`")
        lines.append("")

    # DUPLICATES
    if n_dup:
        lines.append(f"## DUPLICATE — {n_dup} near-duplicate name pairs")
        lines.append("")
        lines.append("Similarity ≥ {:.0%} on the raw name string. Resolve by merging,".format(DUPLICATE_EDIT_RATIO_THRESHOLD))
        lines.append("renaming one, or establishing a clear scope boundary.")
        lines.append("")
        for a, b, ratio in result["duplicates"]:
            lines.append(f"- `{a}` ↔ `{b}` — similarity {ratio}")
        lines.append("")

    # STALE
    if n_stale:
        lines.append(f"## STALE — {n_stale} skills > {STALE_DAYS} days since last edit")
        lines.append("")
        lines.append("Not marked `status: active`. Review for removal or promotion. Usage-based")
        lines.append("staleness detection (N queries where this skill scored > 0) requires")
        lines.append("injection-level telemetry not yet emitted — see #334 follow-ups.")
        lines.append("")
        lines.append("| name | age (days) | status |")
        lines.append("|---|---|---|")
        for item in sorted(result["stale"], key=lambda x: -x["age_days"]):
            lines.append(f"| `{item['name']}` | {item['age_days']} | {item['status']} |")
        lines.append("")

    # `n_phantom` belongs in this condition and was missing from it until
    # 2026-09-11. It is computed above, printed in the table, and rendered as
    # its own section — so a night whose ONLY defect was skills naming tools
    # that do not exist produced a report reading `| PHANTOM_TOOL … | **1** |`
    # in the table and `## ✅ Clean — All skills pass lint` in the body, which
    # is the one category a reader most needs to not miss: a skill naming a
    # tool that does not exist is actively harmful, because the model calls it,
    # gets an unknown-tool error, and takes whatever fallback the skill listed.
    # That is how `web_search`/`WebSearch` taught it to shell out to `curl` for
    # every web lookup on 2026-09-04. Found by task #70's own calibration pass
    # (2026-09-10, finding A1), which proved it by calling `render_report()`
    # with a phantom-only result.
    # MISSING_SCRIPT — #874. The corpus-wide assertion lives in the test suite
    # (`test_no_shipped_skill_names_an_absent_repo_script`), so a row whose path is
    # not listed in `KNOWN_ABSENT_SCRIPTS` has already turned the suite red; this
    # section is where a reader sees WHICH skill and WHICH path, which a non-empty
    # dict in an assert message says only once.
    if n_scripts:
        lines.append(f"## MISSING_SCRIPT — {n_scripts} skill(s) naming a repo script absent from the tree")
        lines.append("")
        lines.append("This is the gap that let a skill ship a day pointing at a script its")
        lines.append("round never committed. Land the script, or drop the citation.")
        lines.append("")
        lines.append("| skill | absent path(s) |")
        lines.append("|---|---|")
        for item in result["missing_script"]:
            paths = ", ".join(f"`{s['path']}`" for s in item["scripts"])
            lines.append(f"| `{item['name']}` | {paths} |")
        lines.append("")

    # INJECTION_PATTERN — #677. The corpus-wide assertion lives in
    # `tests/test_skill_lint_gates.py`, so a hit that is not in INJECTION_ALLOWLIST
    # has already turned the suite red; this section is the audit surface that shows
    # WHICH line of WHICH skill matched WHICH rule, quoted, plus the reason each
    # permitted hit carries. Allow-listed hits print too: the permit is on the gate,
    # not on being seen, so an entry cannot outlive the line it was written for
    # without someone reading the line out of the report.
    if n_inj:
        lines.append(f"## {INJECTION_CATEGORY} — {n_inj} skill(s) whose body matches an injection-shaped rule")
        lines.append("")
        lines.append("This is a review queue, not a detector: a regex over English misses")
        lines.append("paraphrase, so zero here means 'no match', never 'no risk'. The runtime")
        lines.append("measurement stays backlog #590.")
        lines.append("")
        lines.append("| skill | rule | line | quoted match | allow-list reason |")
        lines.append("|---|---|---|---|---|")
        for finding in injection:
            for hit in finding["hits"]:
                quoted = hit["line"].strip().replace("|", "\\|")
                reason = hit["allow_reason"].replace("|", "\\|") or "*(not allowed: the gate fails on this line)*"
                lines.append(f"| `{finding['name']}` | `{hit['rule']}` "
                             f"| {hit['line_no']} | `{quoted[:160]}` | {reason} |")
        lines.append("")

    if not (n_dead or n_missing or n_drift or n_dup or n_stale or n_phantom
            or n_scripts or n_inj):
        blind = untrustworthy_categories()
        if blind:
            # Every count is 0 and some of them could not have been anything
            # else. "All skills pass lint" was what the report said on
            # 2026-09-18 with DRIFT blind to 103 of 194 descriptions, so the
            # verdict names the categories whose zero is not a finding.
            lines.append("## No findings on the checks that can fail")
            lines.append("")
            lines.append("Every count is 0, but this is not a clean bill: "
                         f"{', '.join(blind)} cannot detect anything today, so "
                         "their zeros are not findings. Read them as:")
            lines.append("")
            for category in blind:
                lines.append(f"- **{category}** — 0 is not trustworthy: {CATEGORY_TRUST[category][1]}")
            lines.append("")
        else:
            lines.append("## ✅ Clean")
            lines.append("")
            lines.append("All skills pass lint. No advisories.")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    result = lint()
    # Git, so here and not in `lint()`: a test points `lint` at a temp root with
    # no history. The sample lives in the vault root the report is written into.
    vault = REPORT_PATH.parent.parent
    if (vault / ".git").exists() and "size" in result:
        result["size"]["spill_delta"] = spill_delta(vault)
    report = render_report(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    # Also emit a compact JSON alongside for tooling consumers.
    json_path = REPORT_PATH.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # Counted here, not read out of `render_report`: that is a different
    # function's local, and until this line was fixed the totals line raised
    # NameError on every run that got this far — the one signal the new rule
    # emits was the one that could not print.
    n_scripts = len(result.get("missing_script", []))

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {json_path}")
    # .get, like every count on the Totals line below: a caller that hands
    # `lint` a stub result must still get its totals printed.
    size = result.get("size")
    if size:
        print(f"Size: over_cap={size['over_cap']}/{size['count']} (>{MAX_BODY_LINES} lines), "
              f"p50={size['p50_lines']} p90={size['p90_lines']} max={size['max_lines']}")
    # `phantom` was absent from this line as well, same class of gap one category
    # earlier: the count reached the report and not the stdout the nightly job's
    # log keeps, so `web_search`-shaped breakage had no line to grep.
    print(f"Totals: total={result['total']}, dead={len(result.get('dead', []))}, "
          f"missing_desc={len(result.get('missing_desc', []))}, "
          f"drift={len(result.get('drift', []))}, dup={len(result.get('duplicates', []))}, "
          f"stale={len(result.get('stale', []))}, "
          f"phantom={len(result.get('phantom', []))}, "
          f"missing_script={n_scripts}, "
          f"injection={len(result.get('injection', []))}, "
          f"machine_written={(result.get('authorship') or {}).get('machine_written', 0)}, "
          f"unrecorded_author={len((result.get('authorship') or {}).get('unrecorded', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
