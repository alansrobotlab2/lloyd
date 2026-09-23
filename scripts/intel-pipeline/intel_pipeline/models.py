"""Data models for the intelligence pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


# ── which outcome produced a ScoredItem's relevance (backlog #1380) ──────────
#
# `relevance` alone cannot say this, and the vault writer had to be able to ask.
# No topic in `interests.md` sets `**Weight:**`, so the loader's 1.0 default makes
# `scoring._keyword_fallback` a constant 10: measured on 2026-09-22, 218 of the 258
# stage-1 survivors were never asked because the 40-call stage-2 budget was spent,
# every one of them carried relevance 10, and 101 were written to the vault — while
# `vault_writer.RELEVANCE_FLOOR = 4` held 19 items, all of them model-graded. The
# floor is arithmetically inert on an ungraded item (it scores 10 or 1, never
# anything between), so what the writer needs is the cause of the number, not a
# different number.
GRADE_MODEL = "model"                       # stage 2 returned a usable grade
GRADE_NO_USABLE_GRADE = "no_usable_grade"   # asked, and nothing usable came back
GRADE_CALL_CAP = "call_cap"                 # eligible, but LLM_MAX_CALLS came first
GRADE_KEYWORD = "keyword"                   # never eligible: engine off, or no match


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
    grade_source: str = GRADE_KEYWORD  # see the GRADE_* vocabulary above

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
            category=data.get("category", ""),
            # A feed file written before #1380 carries no cause, and absence must
            # not read as "cap-refused": defaulting to GRADE_CALL_CAP would turn a
            # re-run of `--write` over an old `intel-<date>.jsonl` into a
            # zero-write day, which is the failure this change must never cause.
            grade_source=data.get("grade_source", GRADE_KEYWORD)
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "relevance": self.relevance,
            "urgency": self.urgency,
            "why": self.why,
            "projects": self.projects,
            "category": self.category,
            # Persisted because the writer is a separate invocation: it reads this
            # JSONL back in `load_scored_items`, so a cause that lives only in the
            # scoring process would be absent exactly where the refusal has to act.
            "grade_source": self.grade_source
        })
        return base
    
    @classmethod
    def from_json(cls, json_str: str) -> "ScoredItem":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
