#!/usr/bin/env python3
"""
Lloyd MCP Server: Skills — search and read skill definitions.

Tools: skills_search, skills_read

Skills live in directories under ~/obsidian/skills and ~/lloyd/skills.
Each skill is a folder containing a SKILL.md with YAML frontmatter
(name, description, category, tags) followed by the skill body.
"""

import json
import re
from pathlib import Path
from typing import Optional

import yaml
from mcp.server import Server
from mcp.types import Tool, TextContent

# Skill-side query stopwords moved to agent_mcp._shared (#340 P3 cleanup).
# Local alias preserves the existing internal name `_QUERY_STOPWORDS` so
# the rest of skills.py doesn't change.
from agent_mcp._shared import _SKILLS_QUERY_STOPWORDS as _QUERY_STOPWORDS

# ── Constants ─────────────────────────────────────────────────────────────────

SKILLS_DIRS = [
    Path.home() / "obsidian" / "skills",
    Path(__file__).parent.parent / "skills",
]

# Frontmatter `status:` values that quarantine a skill — it stays on disk
# (and in git history) but is excluded from retrieval entirely. This is the
# lever for pulling a misbehaving skill out of circulation without deleting
# it (e.g. auto-generated bash-runbook skills that the model echoes verbatim
# instead of acting on). The norm is `status: active`; anything in this set
# is skipped. Comparison is case-insensitive on the stripped value.
_QUARANTINE_STATUSES = {"inactive", "archived", "disabled", "retired", "quarantined"}

app = Server("lloyd-skills")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text). Body is everything after the closing ---."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm_text = content[3:end]
    body = content[end + 4:].strip()
    try:
        fm = yaml.safe_load(fm_text) or {}
    except Exception:
        fm = {}
    return fm, body


def _load_skill(skill_dir: Path) -> Optional[dict]:
    """Load and parse a single skill directory. Returns None if no SKILL.md."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None
    try:
        content = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fm, body = _parse_frontmatter(content)
    status = str(fm.get("status", "") or "").strip().lower()
    if status in _QUARANTINE_STATUSES:
        # Quarantined — present on disk but pulled from retrieval.
        return None
    return {
        "name": skill_dir.name,
        "description": fm.get("description", ""),
        "category": fm.get("category", ""),
        "tags": fm.get("tags") or [],
        "body": body,
        "raw": content,
        "path": skill_file,
    }


def _iter_skills():
    """Yield loaded skill dicts from all skill directories."""
    seen = set()
    for skills_dir in SKILLS_DIRS:
        if not skills_dir.exists():
            continue
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in seen:
                continue
            skill = _load_skill(entry)
            if skill:
                seen.add(entry.name)
                yield skill


# Suffixes we collapse for matching. "systems" → "system", "services" → "service",
# "checks" → "check", "tasks" → "task". Conservative: only strip if the result is
# still a real-looking word (≥ 3 chars) and the original isn't already an
# exception (`news`, `bus`, `gas`, `os`, `is`, `us`, `ss`-endings).
_STEM_SKIP = {"news", "bus", "gas", "lens", "kids", "his", "its", "yes", "this", "thus", "plus", "loss", "boss"}


def _stem(token: str) -> str:
    """Cheap morphology collapse for skill-matching only.

    Folds simple English plurals so a query token ("systems") matches a skill
    token ("system"). Not a real stemmer — this is a single-suffix rule tuned
    to fix the prefetch failure mode where "full systems check" picked
    `claude-sdk-check` over `system-health-check` because "systems" ≠ "system".

    Rules (applied in order, first match wins):
      - len ≤ 3: return as-is (too short to safely strip)
      - in _STEM_SKIP: return as-is
      - ends in 'ies' (len > 4): 'ies' → 'y'  (queries → query)
      - ends in 'sses': strip 'es'  (passes → pass)
      - ends in 'ss', 'us', 'is', 'os': return as-is
      - ends in 's': strip 's'  (systems → system)
      - else: return as-is
    """
    if len(token) <= 3 or token in _STEM_SKIP:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith(("ss", "us", "is", "os")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, plural-collapsed for stable matching.

    Both query side and skill side go through stemming so "systems" in a
    query matches "system" in a skill name (and vice versa).
    """
    raw = re.findall(r"\b\w+\b", text.lower())
    return {_stem(t) for t in raw}


def _query_tokens(text: str) -> set[str]:
    """Tokenize a *query* string: lowercase words, stopwords dropped, len ≥ 2.

    Use this for the query side of skill matching. Both sides are stemmed
    via `_tokenize`; the asymmetry vs. skill-side is the stopword filter —
    a query like "lets dig into 311" should not fire skills just because
    "lets", "dig", and "into" appear in arbitrary skill bodies.
    """
    return {t for t in _tokenize(text) if t not in _QUERY_STOPWORDS and len(t) > 1}


def _excerpt(body: str, query_tokens: set[str], max_len: int = 200) -> str:
    """Find the first paragraph containing a query token and return a trimmed excerpt."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
    for para in paragraphs:
        if _tokenize(para) & query_tokens:
            return para[:max_len] + ("…" if len(para) > max_len else "")
    # Fallback: first paragraph
    if paragraphs:
        p = paragraphs[0]
        return p[:max_len] + ("…" if len(p) > max_len else "")
    return ""


# Body-hit cap: with 223 skills, each ~5-15KB of body text, a 5-token query
# will coincidentally match body words in half the skills. Cap the body
# contribution so a skill can't qualify on body-noise alone.
_BODY_HITS_CAP = 4


def _score_skill(skill: dict, query_tokens: set[str],
                 require_metadata_hit: bool = True) -> float:
    """Score a skill against query tokens. Higher = more relevant.

    If `require_metadata_hit` is True (default), a skill with zero
    name/desc/tag overlap scores 0.0 regardless of body overlap. This
    prevents the "12× powerpoint on graph-classifier work" failure mode
    where generic English stopwords in queries bag-of-words-match arbitrary
    skill bodies.

    Body hits are capped (see `_BODY_HITS_CAP`) and weighted to be a
    tiebreaker, not a qualifier.
    """
    name_tokens = _tokenize(skill["name"].replace("-", " "))
    desc_tokens = _tokenize(skill["description"])
    tag_tokens = _tokenize(" ".join(skill["tags"]))
    body_tokens = _tokenize(skill["body"])

    name_hits = len(query_tokens & name_tokens)
    desc_hits = len(query_tokens & desc_tokens)
    tag_hits = len(query_tokens & tag_tokens)
    body_hits = min(len(query_tokens & body_tokens), _BODY_HITS_CAP)

    if require_metadata_hit and (name_hits + desc_hits + tag_hits) == 0:
        return 0.0

    return name_hits * 3.0 + desc_hits * 2.0 + tag_hits * 1.5 + body_hits * 0.3


# ── Tool handlers ─────────────────────────────────────────────────────────────

def _skills_search(params: dict) -> str:
    query = params.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required", "results": []})
    max_results = int(params.get("max_results", 10))
    query_tokens = _query_tokens(query)

    scored = []
    for skill in _iter_skills():
        score = _score_skill(skill, query_tokens, require_metadata_hit=True)
        if score > 0:
            scored.append((score, skill))

    scored.sort(key=lambda x: -x[0])

    results = []
    for score, skill in scored[:max_results]:
        results.append({
            "name": skill["name"],
            "description": skill["description"],
            "category": skill["category"],
            "tags": skill["tags"],
            "excerpt": _excerpt(skill["body"], query_tokens),
            "score": round(score, 2),
        })

    return json.dumps({"query": query, "results": results, "total": len(scored)})


def _skills_read(params: dict) -> str:
    name = params.get("name", "").strip()
    if not name:
        return json.dumps({"error": "name is required"})
    for skills_dir in SKILLS_DIRS:
        skill_dir = skills_dir / name
        skill = _load_skill(skill_dir)
        if skill:
            return json.dumps({"name": skill["name"], "content": skill["raw"]})
    return json.dumps({"error": f"Skill not found: {name}"})


# ── MCP registration ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="skills_search",
            description=(
                "Search available skills by keyword. Searches skill names, descriptions, "
                "tags, and body content. Returns ranked results with name, description, "
                "and a body excerpt. Use this to discover which skill to apply to a task."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for"},
                    "max_results": {"type": "integer", "description": "Max results to return (default 10)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="skills_read",
            description=(
                "Read the full SKILL.md content for a named skill. Use after skills_search "
                "to get the complete instructions for a specific skill."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill directory name (e.g. 'research-agent')"},
                },
                "required": ["name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handlers = {
        "skills_search": _skills_search,
        "skills_read": _skills_read,
    }
    handler = handlers.get(name)
    if handler:
        return [TextContent(type="text", text=handler(arguments))]
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
