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
    # When the SOURCE says the thing was published, as distinct from when we found
    # it. YouTube's Atom feed carries `atom:published` on every entry and the
    # scanner parsed it into a local (backlog #1379) with nowhere to put it, so the
    # vault writer could not tell a video from this morning from one from 453 days
    # ago and appended both under today's heading. Empty means the source served no
    # date — which is a real state (the GitHub scanner has no such field), not a
    # failure, and the age gate must treat it as inside the window.
    published: str = ""
    # The channel's description as the feed served it, BEFORE `strip_link_footer` and
    # `clip_body` touch it (backlog #2241). `summary` is what the vault gets, and a
    # description that is nothing but a promotional link block strips to `""` by
    # design — a link farm is not knowledge prose, and `strip_link_footer`'s own
    # docstring hands the caller that decision. The YouTube scanner took the writer's
    # side of it and left the stage-1 gate reading `f"{title} {summary}"`, so such a
    # video was judged on its title alone and the profile keyword sitting in the
    # channel's own label was never in front of the gate. This field exists for that
    # gate and nothing else: `vault_writer` must not read it, and stage 2's three other
    # readers of `item.summary` (`_score_prompt`, `_keyword_fallback`,
    # `match_projects`) were left reading the stored summary on purpose — widening
    # them is an open question on #2241, not a change made here.
    #
    # Empty means the scanner carries no such text: every GitHub row, and every YouTube
    # row written before this field existed, which is what `stage1_text` reads as
    # "gate on the stored summary" — the behaviour those rows have today.
    gate_description: str = ""

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
            source_tags=data.get("source_tags", []),
            # A row written before this field existed has no key at all, and
            # absence must read as "the source gave no date" — never as today, and
            # never as a KeyError in a re-run of an old day.
            published=data.get("published", ""),
            # Same rule for the same reason (#2241): the day files already on disk
            # predate this key, and a `--score --date` replay of one of them has to
            # gate on title + summary rather than die on a missing key.
            gate_description=data.get("gate_description", "")
        )

    def stage1_text(self) -> str:
        """The text `stage1_filter` scores: the title, plus the widest copy we have.

        `gate_description` where a scanner carries one, otherwise the stored summary,
        so a row that predates the field — or a feed whose scanner never had pre-strip
        text to keep — is gated exactly as it is today. This is the gate's method and
        not the writer's: what reaches `knowledge/` stays `summary`, stripped, because
        the strip that emptied it is a decision about publishing, not about matching.
        """
        return f"{self.title} {self.gate_description or self.summary}"

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
            "source_tags": self.source_tags,
            "published": self.published,
            # Persisted because scanning and scoring are separate invocations joined by
            # `raw/<date>.jsonl` (`__main__.py:129` re-reads it): gate text that lived
            # only in the scan would be absent exactly where the gate runs.
            "gate_description": self.gate_description
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
            # The scoring stage and the write stage are separate processes joined by
            # `intel-<date>.jsonl`, so a date that survives the scanner but not this
            # rebuild is invisible exactly where the age gate has to act (#1379).
            published=data.get("published", ""),
            # A feed file written before #1380 carries no cause, and absence must
            # not read as "cap-refused": defaulting to GRADE_CALL_CAP would turn a
            # re-run of `--write` over an old `intel-<date>.jsonl` into a
            # zero-write day, which is the failure this change must never cause.
            grade_source=data.get("grade_source", GRADE_KEYWORD),
            # `to_dict` inherits this key from `FeedItem`, so `from_dict` has to read
            # it back or the round trip silently loses a field it just wrote. Nothing
            # downstream reads it: the writer's body comes from `summary` (#2241).
            gate_description=data.get("gate_description", "")
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
