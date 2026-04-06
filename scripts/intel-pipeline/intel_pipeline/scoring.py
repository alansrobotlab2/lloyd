"""Two-stage scoring pipeline for intelligence items."""

from datetime import datetime
from typing import List, Optional, Dict, Any

from .models import FeedItem, ScoredItem
from .profile import load_profile, keyword_match, keyword_score, get_all_projects


def stage1_filter(
    items: List[FeedItem],
    profile: dict
) -> List[FeedItem]:
    """
    Stage 1: Filter items based on interest profile.
    
    Args:
        items: List of FeedItem objects
        profile: Interest profile dictionary
    
    Returns:
        Filtered list of FeedItem objects
    """
    filtered_items = []
    
    for item in items:
        # Combine title and summary for matching
        text = f"{item.title} {item.summary}"
        
        # Check against profile keywords
        matched = keyword_match(text, profile)
        
        # If matched any topics, keep the item
        if matched or item.source_tags:
            filtered_items.append(item)
    
    return filtered_items


def stage2_score(
    items: List[FeedItem],
    profile: dict
) -> List[ScoredItem]:
    """
    Stage 2: Score filtered items.
    
    Args:
        items: List of FeedItem objects
        profile: Interest profile dictionary
    
    Returns:
        List of ScoredItem objects
    """
    scored_items = []
    all_projects = get_all_projects(profile)
    
    for item in items:
        # Combine title and summary for matching
        text = f"{item.title} {item.summary}"
        
        # Get matched topics
        matched_topics = keyword_match(text, profile)
        
        # Calculate relevance score (1-10) based on max weight
        if matched_topics:
            max_weight = max(topic["weight"] for topic in matched_topics)
            relevance = max(1, min(10, int(round(max_weight * 10))))
        else:
            relevance = 1
        
        # Determine urgency
        urgency = determine_urgency(relevance)
        
        # Generate why explanation
        why = generate_why(item, matched_topics)
        
        # Match to projects
        matched_projects = match_projects(item, all_projects)
        
        # Determine category
        category = determine_category(matched_topics)
        
        scored_item = ScoredItem(
            id=item.id,
            source=item.source,
            title=item.title,
            url=item.url,
            summary=item.summary,
            discovered_at=item.discovered_at,
            authors=item.authors,
            source_tags=item.source_tags,
            relevance=relevance,
            urgency=urgency,
            why=why,
            projects=matched_projects,
            category=category
        )
        scored_items.append(scored_item)
    
    return scored_items


def determine_urgency(relevance: int) -> str:
    """Determine urgency level based on relevance score."""
    if relevance >= 8:
        return "urgent"
    elif relevance >= 6:
        return "morning"
    elif relevance >= 4:
        return "weekly"
    else:
        return "low"


def generate_why(item: FeedItem, matched_topics: List[Dict]) -> str:
    """Generate explanation for why this item is interesting."""
    if not matched_topics:
        return "Matches general interests"
    
    # Get all matched keywords
    all_matches = []
    for topic in matched_topics:
        all_matches.extend(topic.get("matched_keywords", []))
    
    unique_matches = list(set(all_matches))[:5]  # Limit to 5 matches
    return f"Matches: {', '.join(unique_matches)}"


def match_projects(item: FeedItem, projects: List[str]) -> List[str]:
    """Match item to projects."""
    text = f"{item.title} {item.summary}".lower()
    matched = []
    
    for project in projects:
        if project.lower() in text:
            matched.append(project)
    
    return matched


def determine_category(matched_topics: List[Dict]) -> str:
    """Determine the primary category for an item."""
    if matched_topics:
        # Return the highest weighted topic name
        return max(matched_topics, key=lambda x: x["weight"])["name"]
    return "general"


def run_scoring_pipeline(
    items: List[FeedItem],
    profile: Optional[dict] = None
) -> List[ScoredItem]:
    """
    Run the full two-stage scoring pipeline.
    
    Args:
        items: List of FeedItem objects
        profile: Interest profile (optional, loads default if not provided)
    
    Returns:
        List of ScoredItem objects
    """
    if profile is None:
        profile = load_profile()
    
    # Stage 1: Filter
    filtered = stage1_filter(items, profile)
    
    # Stage 2: Score
    scored = stage2_score(filtered, profile)
    
    return scored
