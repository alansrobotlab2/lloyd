"""Interest profile loader with keyword matching."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Vault root interests file
PROFILE_FILE = Path.home() / "obsidian/interests.md"

# Optional per-topic field block, one field per line, bullet optional:
#   ## Robotics
#   **Weight:** 0.7
#   **Projects:** Alfie
#   humanoid, actuator, servo
# Until 2026-09-11 none of these were read (backlog #570): the loader hardcoded
# weight 1.0 and projects [] for every topic, which made the 1-10 relevance
# scale mathematically binary and match_projects() permanently empty.
_FIELD_LINE = re.compile(
    r'^\s*[-*]?\s*\*\*(Weight|Projects|Keywords|Depth)\s*:?\*\*\s*:?\s*(.*?)\s*$',
    re.IGNORECASE)


def slugify(text: str) -> str:
    """Convert text to slug format (lowercase, spaces to hyphens)."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower().strip()).strip('-')


def _split_list(value: str) -> List[str]:
    return [v.strip() for v in re.split(r'[,;]', value) if v.strip()]


def _parse_topic_body(body: str) -> Dict[str, Any]:
    """Split a topic body into its declared fields and its leftover keywords.

    Field lines are consumed, never kept as keywords: a `**Weight:** 0.7` line
    passing through as a keyword would displace a real one and match nothing.
    """
    fields: Dict[str, Any] = {}
    leftover: List[str] = []

    for line in body.split("\n"):
        m = _FIELD_LINE.match(line)
        if not m:
            leftover.append(line)
            continue
        key, value = m.group(1).lower(), m.group(2).strip()
        if key == "weight":
            try:
                fields["weight"] = max(0.0, min(1.0, float(value)))
            except ValueError:
                pass  # an unparseable weight leaves the default, not a crash
        elif key == "projects":
            fields["projects"] = _split_list(value)
        elif key == "keywords":
            fields["keywords"] = _split_list(value)
        elif key == "depth":
            fields["depth"] = value

    if "keywords" not in fields:
        bare = [ln.strip().lstrip("-* ").strip() for ln in leftover]
        fields["keywords"] = _split_list(",".join(b for b in bare if b))
    fields.setdefault("weight", 1.0)
    fields.setdefault("projects", [])
    fields.setdefault("depth", "deep")
    return fields


def load_profile(path: Optional[str] = None) -> dict:
    """Load interest profile from markdown file.
    
    Format: H2 headings as topic names, comma-separated keywords as body text.
    A topic may also declare `**Weight:**` (0-1), `**Projects:**`,
    `**Keywords:**` and `**Depth:**` as one-per-line fields.
    
    Example:
        ## Robotics
        **Weight:** 0.7
        **Projects:** Alfie
        humanoid, actuator, servo, DOF, gait
        
        ## AI & LLMs
        qwen, vllm, quantization
    """
    profile_path = Path(path) if path else PROFILE_FILE
    
    if not profile_path.exists():
        return {"topics": [], "all_keywords": [], "all_projects": []}
    
    content = profile_path.read_text()
    topics = []
    
    # Split by H2 headings (## Topic Name)
    h2_pattern = r'^## (.+?)$'
    h2_matches = list(re.finditer(h2_pattern, content, re.MULTILINE))
    
    for i, match in enumerate(h2_matches):
        topic_name = match.group(1).strip()
        
        # Get content between this H2 and the next (or end)
        start = match.end()
        end = h2_matches[i + 1].start() if i + 1 < len(h2_matches) else len(content)
        body = content[start:end].strip()
        
        # Parse declared fields first, then whatever keyword text is left over
        parsed = _parse_topic_body(body)
        keywords = parsed["keywords"]
        
        if keywords:
            topics.append({
                "name": slugify(topic_name),
                "weight": parsed["weight"],
                "keywords": keywords,
                "projects": parsed["projects"],
                "depth": parsed["depth"]
            })
    
    all_kw = []
    for t in topics:
        all_kw.extend(t["keywords"])
    
    all_projects = []
    for t in topics:
        for p in t["projects"]:
            if p not in all_projects:
                all_projects.append(p)
    
    return {"topics": topics, "all_keywords": all_kw, "all_projects": all_projects}


def get_all_keywords(profile: dict) -> List[str]:
    """Get all keywords from all topics."""
    keywords = []
    for topic in profile.get("topics", []):
        keywords.extend(topic.get("keywords", []))
    return keywords


def get_topics(profile: dict) -> List[dict]:
    """Get all topics from the profile."""
    return profile.get("topics", [])


def match_keywords(text: str, keywords: List[str]) -> List[str]:
    """Match keywords against text, return matching keywords."""
    if not text or not keywords:
        return []
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def keyword_match(text: str, profile: dict) -> List[Dict[str, Any]]:
    """Match text against topic keywords, return matched topics with weights."""
    matched = []
    for topic in profile.get("topics", []):
        matches = match_keywords(text, topic.get("keywords", []))
        if matches:
            matched.append({
                "name": topic["name"],
                "weight": topic.get("weight", 1.0),
                "matched_keywords": matches
            })
    return matched


def keyword_score(text: str, profile: dict) -> float:
    """Calculate keyword score for text (0.0 to 1.0). Returns max weighted topic match."""
    matched = keyword_match(text, profile)
    return max((t["weight"] for t in matched), default=0.0)


def get_interest_profile() -> dict:
    """Get the full interest profile."""
    return load_profile()


def get_all_projects(profile: dict) -> List[str]:
    """Get all projects from the profile."""
    projects = []
    for topic in profile.get("topics", []):
        projects.extend(topic.get("projects", []))
    return projects
