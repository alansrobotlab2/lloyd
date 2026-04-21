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

# ── Constants ─────────────────────────────────────────────────────────────────

SKILLS_DIRS = [
    Path.home() / "obsidian" / "skills",
    Path(__file__).parent.parent / "skills",
]

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


# Conversational noise + function words — stripped from *query* side before
# scoring. Skill-side tokenization is unchanged: we still match against full
# name/desc/tag/body text. The asymmetry is deliberate — a query like "lets
# dig into 311" should not fire skills just because "lets", "dig", and
# "into" appear in arbitrary skill bodies.
_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "his", "her",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "between",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "just", "also", "very", "really", "quite", "too", "much",
    "ok", "okay", "yeah", "yes", "nah", "sure", "right",
    "lets", "let", "go", "going", "get", "got", "getting",
    "want", "wants", "wanted", "know", "knows", "knew",
    "think", "thinks", "thought", "look", "looking", "looked",
    "take", "takes", "took", "make", "makes", "made",
    "now", "some", "any", "all", "each", "every", "both",
    "up", "out", "over", "down", "off", "away",
    "here", "there", "thing", "things", "stuff",
    "left", "done", "next", "back", "ready", "still", "already",
    "tell", "show", "give", "put", "run", "running", "ran",
    "come", "came", "see", "saw", "seen", "say", "said",
    "try", "tried", "use", "used", "using",
    "start", "started", "stop", "stopped", "keep", "kept",
    "set", "well", "good",
    "bit", "lot", "way", "something", "anything", "everything",
    "like", "first", "last", "new", "old", "one", "two",
    "dig", "really",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _query_tokens(text: str) -> set[str]:
    """Tokenize a *query* string: lowercase words, stopwords dropped, len ≥ 2.

    Use this for the query side of skill matching. Skill-side tokenization
    stays with plain `_tokenize` so we keep recall on legitimate substantive
    words that happen to appear in skill content.
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
