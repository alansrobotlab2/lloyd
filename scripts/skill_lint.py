#!/usr/bin/env python3
"""
Skill-lint — advisory quality sweep over ~/obsidian/skills/*/SKILL.md.

Writes `~/obsidian/autonomy/skill-lint-report.md` with six categories:
  1. DEAD         — unparseable frontmatter or desc+tags both empty (never fires)
  2. MISSING_DESC — has tags but no description (fires via tag/name only)
  3. DRIFT        — description present but output-framed not trigger-framed
  4. DUPLICATE    — near-duplicate skill names (ranking noise)
  5. STALE        — mtime > 90 days and status != active (candidate for removal)
  6. PHANTOM_TOOL — names a tool the aggregator does not advertise

Advisory only. No automatic deletion or rewrites. Exit 0 always (so nightly
pipeline doesn't fail on lint findings).

Origin: Task #334. Methodology documented in that task's description.
"""
from __future__ import annotations

import datetime as dt
import difflib
import json
import re
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    print("ERROR: pyyaml not available. Run from .venvs/lloyd.", file=sys.stderr)
    sys.exit(2)


SKILLS_DIR = Path.home() / "obsidian" / "skills"
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
        # Heuristic guard: imperative verbs are usually the FIRST token AND
        # the second token is typically a noun or article, not another verb.
        # We don't try to verify — calibration check below catches over-pass.
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


# ── Runner ────────────────────────────────────────────────────────────────────

def lint() -> dict:
    if not SKILLS_DIR.exists():
        return {
            "error": f"skills dir missing: {SKILLS_DIR}",
            "dead": [], "drift": [], "duplicates": [], "stale": [],
            "total": 0, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }

    dead: list[dict] = []
    missing_desc: list[dict] = []
    drift: list[dict] = []
    stale: list[dict] = []
    phantom: list[dict] = []
    skills: list[tuple[str, str]] = []
    total = 0

    for entry in sorted(SKILLS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.exists():
            continue
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

        fm, _body, yaml_err = parse_frontmatter(content)

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
        "dead": dead,
        "missing_desc": missing_desc,
        "drift": drift,
        "duplicates": duplicates,
        "stale": stale,
        "phantom": phantom,
    }


# ── Report writer ─────────────────────────────────────────────────────────────

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

    lines.append(f"# Skill Lint Report — {ts}")
    lines.append("")
    lines.append(f"Scanned **{total}** skills in `~/obsidian/skills/`.")
    lines.append("")
    lines.append("| category | count | action |")
    lines.append("|---|---|---|")
    lines.append(f"| DEAD (never fires) | **{n_dead}** | fix frontmatter or remove |")
    lines.append(f"| MISSING_DESC (tags only, no description) | **{n_missing}** | add trigger-condition description |")
    lines.append(f"| DRIFT (output-framed description) | **{n_drift}** | rewrite first sentence in trigger-condition form |")
    lines.append(f"| DUPLICATE (near-duplicate names) | **{n_dup}** | resolve ownership, merge, or rename |")
    lines.append(f"| STALE (>{STALE_DAYS}d mtime, status ≠ active) | **{n_stale}** | review for removal |")
    lines.append(f"| PHANTOM_TOOL (names a tool that does not exist) | **{n_phantom}** | replace with the real tool name |")
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

    if not (n_dead or n_missing or n_drift or n_dup or n_stale):
        lines.append("## ✅ Clean")
        lines.append("")
        lines.append("All skills pass lint. No advisories.")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    result = lint()
    report = render_report(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    # Also emit a compact JSON alongside for tooling consumers.
    json_path = REPORT_PATH.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {json_path}")
    print(f"Totals: total={result['total']}, dead={len(result.get('dead', []))}, "
          f"missing_desc={len(result.get('missing_desc', []))}, "
          f"drift={len(result.get('drift', []))}, dup={len(result.get('duplicates', []))}, "
          f"stale={len(result.get('stale', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
