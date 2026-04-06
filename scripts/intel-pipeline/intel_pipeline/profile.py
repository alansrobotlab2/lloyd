"""Interest profile loader with keyword matching."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Vault root interests file
PROFILE_FILE = Path.home() / "obsidian/interests.md"


def slugify(text: str) -> str:
    """Convert text to slug format (lowercase, spaces to hyphens)."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower().strip()).strip('-')


def load_profile(path: Optional[str] = None) -> dict:
    """Load interest profile from markdown file.
    
    Format: H2 headings as topic names, comma-separated keywords as body text.
    
    Example:
        ## Robotics
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
        
        # Parse comma-separated keywords from body text
        keywords = [kw.strip() for kw in body.split(",") if kw.strip()]
        
        if keywords:
            topics.append({
                "name": slugify(topic_name),
                "weight": 1.0,
                "keywords": keywords,
                "projects": [],
                "depth": "deep"
            })
    
    all_kw = []
    for t in topics:
        all_kw.extend(t["keywords"])
    
    return {"topics": topics, "all_keywords": all_kw, "all_projects": []}


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
