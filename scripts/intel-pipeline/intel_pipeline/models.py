"""Data models for the intelligence pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class FeedItem:
    """Base item from any feed source."""
    id: str
    source: str
    title: str
    url: str
    summary: str
    discovered_at: str  # ISO-8601 format
    authors: list = field(default_factory=list)
    source_tags: list = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeedItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source=data["source"],
            title=data["title"],
            url=data["url"],
            summary=data.get("summary", ""),
            discovered_at=data["discovered_at"],
            authors=data.get("authors", []),
            source_tags=data.get("source_tags", [])
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "discovered_at": self.discovered_at,
            "authors": self.authors,
            "source_tags": self.source_tags
        }
    
    @classmethod
    def from_json(cls, json_str: str) -> "FeedItem":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ScoredItem(FeedItem):
    """FeedItem with scoring metadata."""
    relevance: int = 1  # 1-10
    urgency: str = "low"  # urgent|morning|weekly|low
    why: str = ""
    projects: list = field(default_factory=list)
    category: str = ""
    
    @classmethod
    def from_dict(cls, data: dict) -> "ScoredItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source=data["source"],
            title=data["title"],
            url=data["url"],
            summary=data.get("summary", ""),
            discovered_at=data["discovered_at"],
            authors=data.get("authors", []),
            source_tags=data.get("source_tags", []),
            relevance=data.get("relevance", 1),
            urgency=data.get("urgency", "low"),
            why=data.get("why", ""),
            projects=data.get("projects", []),
            category=data.get("category", "")
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "relevance": self.relevance,
            "urgency": self.urgency,
            "why": self.why,
            "projects": self.projects,
            "category": self.category
        })
        return base
    
    @classmethod
    def from_json(cls, json_str: str) -> "ScoredItem":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
