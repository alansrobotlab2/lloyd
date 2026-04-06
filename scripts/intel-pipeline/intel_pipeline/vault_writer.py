"""Vault writer module for storing scored items to Obsidian vault."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import ScoredItem
from .profile import load_profile, get_all_keywords, keyword_match


# Paths
VAULT_ROOT = Path.home() / "obsidian"
KNOWLEDGE_DIR = VAULT_ROOT / "knowledge"
SCORED_FEEDS_DIR = VAULT_ROOT / "memory/feeds"
VAULT_WRITTEN_STATE = VAULT_ROOT / "memory/feeds/vault-written.json"


def load_scored_items(date_str: str) -> List[ScoredItem]:
    """Load scored items from JSONL file."""
    scored_path = SCORED_FEEDS_DIR / f"intel-{date_str}.jsonl"
    
    if not scored_path.exists():
        return []
    
    items = []
    with open(scored_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                item = ScoredItem.from_dict(data)
                items.append(item)
    
    return items


def load_written_state() -> Dict:
    """Load state of items already written to vault."""
    if not VAULT_WRITTEN_STATE.exists():
        return {"written": []}
    
    with open(VAULT_WRITTEN_STATE, "r") as f:
        return json.load(f)


def save_written_state(state: Dict):
    """Save state of items written to vault."""
    VAULT_WRITTEN_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(VAULT_WRITTEN_STATE, "w") as f:
        json.dump(state, f, indent=2)


def is_written(item_id: str, state: Dict) -> bool:
    """Check if an item has already been written to vault."""
    return item_id in state.get("written", [])


def mark_written(item_id: str, state: Dict):
    """Mark an item as written to vault."""
    if "written" not in state:
        state["written"] = []
    if item_id not in state["written"]:
        state["written"].append(item_id)


def determine_vault_path(item: ScoredItem, profile: dict) -> Path:
    """Determine the vault path for a scored item."""
    source = item.source.lower()
    category = item.category.lower()
    text = f"{item.title} {item.summary}".lower()
    
    # GitHub items
    if source == "github":
        # Extract repo name from URL or title
        if "github.com" in item.url:
            url_parts = item.url.split("/")
            if len(url_parts) >= 5:
                repo_name = url_parts[4].lower().replace(".git", "")
            else:
                repo_name = "unknown-repo"
        else:
            repo_name = "unknown-repo"
        
        # Determine type from source_tags
        is_pr = "pr" in [t.lower() for t in item.source_tags]
        is_release = "release" in [t.lower() for t in item.source_tags]
        
        if is_release:
            return KNOWLEDGE_DIR / "tools" / repo_name / "releases.md"
        elif is_pr:
            return KNOWLEDGE_DIR / "tools" / repo_name / "prs.md"
        else:
            return KNOWLEDGE_DIR / "tools" / repo_name / "updates.md"
    
    # YouTube items
    elif source == "youtube":
        # Match to interest profile topics
        matched_topics = keyword_match(text, profile)
        if matched_topics:
            # Use highest weighted topic
            topic_name = max(matched_topics, key=lambda x: x["weight"])["name"]
            topic_slug = topic_name.lower().replace("_", "-")
            return KNOWLEDGE_DIR / topic_slug / "youtube-digest.md"
        else:
            return KNOWLEDGE_DIR / "feeds" / "youtube-uncategorized.md"
    
    # arXiv items
    elif source == "arxiv":
        matched_topics = keyword_match(text, profile)
        if matched_topics:
            topic_name = max(matched_topics, key=lambda x: x["weight"])["name"]
            topic_slug = topic_name.lower().replace("_", "-")
            return KNOWLEDGE_DIR / topic_slug / "papers.md"
        else:
            return KNOWLEDGE_DIR / "feeds" / "arxiv-uncategorized.md"
    
    # Hacker News items
    elif source == "hackernews":
        matched_topics = keyword_match(text, profile)
        if matched_topics:
            topic_name = max(matched_topics, key=lambda x: x["weight"])["name"]
            topic_slug = topic_name.lower().replace("_", "-")
            return KNOWLEDGE_DIR / topic_slug / "news.md"
        else:
            return KNOWLEDGE_DIR / "feeds" / "hn-uncategorized.md"
    
    # Fallback
    else:
        return KNOWLEDGE_DIR / "feeds" / "uncategorized.md"


def url_exists_in_file(url: str, file_path: Path) -> bool:
    """Check if a URL already exists in the file (simple dedup)."""
    if not file_path.exists():
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return url in content
    except Exception:
        return False


def write_item_to_vault(item: ScoredItem, profile: dict) -> bool:
    """Write a single scored item to the vault."""
    vault_path = determine_vault_path(item, profile)
    
    # Ensure parent directory exists
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for duplicates
    if url_exists_in_file(item.url, vault_path):
        print(f"  Skipping (already exists): {item.url}")
        return False
    
    # Prepare content
    today = datetime.utcnow().strftime("%Y-%m-%d")
    source = item.source
    relevance = item.relevance
    category = item.category or "general"
    
    # Build frontmatter tags
    tags = ["intel-pipeline", source, category]
    
    # Format content - just the entry, not the full file
    summary = item.summary if item.summary else "(No description)"
    
    content = f"""## {today}

### {item.title}

**Source:** {source} | **Relevance:** {relevance}/10

{summary}

[Link]({item.url})

---

"""
    
    # Append or create file
    if vault_path.exists():
        # Append to existing file
        with open(vault_path, "a") as f:
            f.write(content)
    else:
        # Create new file with frontmatter and header
        header = f"""---
segment: knowledge
type: notes
tags:
{chr(10).join(f'  - {tag}' for tag in tags)}
---

# {category.title()} Updates

"""
        with open(vault_path, "w") as f:
            f.write(header + content)
    
    print(f"  Written: {vault_path.relative_to(VAULT_ROOT)}")
    return True


def write_all_to_vault(date_str: Optional[str] = None) -> int:
    """
    Write all scored items for a date to the vault.
    
    Args:
        date_str: Date string in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Number of items written
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    print(f"\n=== Vault Writer for {date_str} ===\n")
    
    # Load scored items
    items = load_scored_items(date_str)
    if not items:
        print(f"No scored items found for {date_str}")
        return 0
    
    print(f"Loaded {len(items)} scored items")
    
    # Load written state
    written_state = load_written_state()
    
    # Load interest profile
    profile = load_profile()
    
    # Write items
    written_count = 0
    for item in items:
        if is_written(item.id, written_state):
            print(f"  Skipping (already written): {item.id}")
            continue
        
        try:
            if write_item_to_vault(item, profile):
                mark_written(item.id, written_state)
                written_count += 1
        except Exception as e:
            print(f"  Error writing {item.id}: {e}")
    
    # Save written state
    save_written_state(written_state)
    
    print(f"\n=== Vault Write Complete ===")
    print(f"Wrote {written_count} items to vault")
    
    return written_count


if __name__ == "__main__":
    # Example usage
    count = write_all_to_vault()
    print(f"\nTotal items written: {count}")
