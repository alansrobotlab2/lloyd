"""Two-stage scoring pipeline for intelligence items.

Stage 1 is a keyword gate against `~/obsidian/interests.md`. Stage 2 asks the
local model to grade what got through.

The stage-2 model call is the whole point of the file and it did not exist until
2026-09-11 (backlog #570): `stage2_score()` derived relevance from
`max_weight * 10`, every weight was the loader's hardcoded 1.0, and the measured
relevance distribution was exactly {1: 68, 10: 13} over a day — a 1-10 scale
with no value between the endpoints. `interests.md` claimed a non-matching item
"gets ignored" while stage 1 in fact passed 103/103 items because every scanner
attaches `source_tags`. See tests/test_intel_pipeline_scorer.py.
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from typing import Callable, List, Optional, Dict, Any

from .models import FeedItem, ScoredItem
from .profile import load_profile, keyword_match, keyword_score, get_all_projects

# Stage 2 only runs for items the keyword stage already rates above this.
# Matches the design the `intelligence-pipeline` skill documents ("LLM
# relevance scoring, only if keyword score > 0.3"); with the loader now reading
# real weights, 0.3 is a floor an unweighted topic (1.0) clears and a
# non-match (0.0) never does.
LLM_KEYWORD_THRESHOLD = 0.3

# The pipeline runs under a 1800 s task timeout (autonomy task #30) and a model
# call costs seconds. Uncapped, a bad day of items becomes a timeout, which is
# the failure mode the fleet already tracks for other tasks. Items past the cap
# keep their keyword-derived score and say so in the run output.
LLM_MAX_CALLS = 40
LLM_TIMEOUT_SECONDS = 90

# Same endpoint and payload conventions as scripts/youtube_channel_monitor.py
# and scripts/memory/next-gen-memory/fact_extractor.py — stdlib urllib, model
# "primary", thinking suppressed. Deliberately not a new client: this package is
# stdlib + pyyaml by design (see its requirements.txt note in backlog #570).
LLM_URL = os.environ.get("INTEL_LLM_URL",
                         "http://localhost:8096/v1/chat/completions")
LLM_MODEL = os.environ.get("INTEL_LLM_MODEL", "primary")

_SCORE_SYSTEM = (
    "You score how relevant one feed item is to one person's stated interests, "
    "on a 1-10 scale. Think about what they are actually building, not whether "
    "the title contains a buzzword. A generic AI-headline or an unrelated "
    "hardware clip is 1-3 even if it mentions AI. Respond with JSON only."
)


def _score_prompt(item: FeedItem, profile: dict) -> str:
    """Build the stage-2 prompt: the item, the interest profile, active projects."""
    topics = ", ".join(
        f"{t['name']} ({t.get('weight', 1.0)})"
        for t in profile.get("topics", [])
    ) or "(no topics declared)"
    projects = ", ".join(get_all_projects(profile)) or "(none declared)"
    return (
        f"Interests and weights: {topics}\n"
        f"Active projects: {projects}\n\n"
        f"Item source: {item.source}\n"
        f"Item title: {item.title}\n"
        f"Item summary: {(item.summary or '')[:900]}\n\n"
        'Return JSON exactly: {"relevance": <int 1-10>, "why": "<one short sentence>", '
        '"projects": [<matching project names>], "category": "<short-slug topic>"}'
    )


def call_local_llm(prompt: str) -> str:
    """POST to the local model and return its message content.

    Raises on any transport or shape failure — a scoring stage that silently
    returns "" would look like a quiet day, which is the bug this item is about.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SCORE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        # llama.cpp knob; reasoning tokens otherwise eat the whole budget
        "chat_template_kwargs": {"enable_thinking": False},
        # vLLM --scheduling-policy priority: interactive 0, autonomy 1, batch 2
        "priority": 2,
    }
    req = urllib.request.Request(
        LLM_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as resp:
        result = json.loads(resp.read().decode())
    content = (result.get("choices") or [{}])[0].get("message", {}).get("content")
    if not content or not content.strip():
        raise RuntimeError("local model returned empty content")
    return content


def _parse_score_json(text: str) -> Optional[dict]:
    """Pull the JSON object out of a model reply, which may be fenced or padded."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _clamp_relevance(value: Any) -> Optional[int]:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 10:
        return None  # an out-of-range grade is a failed grade, not a clamped one
    return score


def _keyword_fallback(item: FeedItem, profile: dict) -> dict:
    """Keyword-only score, used when the model is unavailable or answered junk."""
    text = f"{item.title} {item.summary}"
    matched_topics = keyword_match(text, profile)
    if matched_topics:
        max_weight = max(topic["weight"] for topic in matched_topics)
        relevance = max(1, min(10, int(round(max_weight * 10))))
    else:
        relevance = 1
    return {
        "relevance": relevance,
        "why": generate_why(item, matched_topics),
        "projects": None,
        "category": None,
        "matched_topics": matched_topics,
    }


def stage1_filter(
    items: List[FeedItem],
    profile: dict
) -> List[FeedItem]:
    """
    Stage 1: keep items whose text matches the interest profile.

    Args:
        items: List of FeedItem objects
        profile: Interest profile dictionary

    Returns:
        Filtered list of FeedItem objects

    The dropped half is the point. Until 2026-09-11 this read
    `if matched or item.source_tags`, and both scanners attach `source_tags` to
    every item they emit — so it returned 103 of 103 and every repo commit
    reached the writer as 10/10 "urgent" on the repo name alone. Repo traffic is
    not an interest signal; `interests.md` says what is.
    """
    filtered_items = []

    for item in items:
        # Combine title and summary for matching
        text = f"{item.title} {item.summary}"

        # Check against profile keywords
        matched = keyword_match(text, profile)

        if matched:
            filtered_items.append(item)

    return filtered_items


def stage2_score(
    items: List[FeedItem],
    profile: dict,
    llm_call: Optional[Callable[[str], str]] = None,
    max_llm_calls: int = LLM_MAX_CALLS,
) -> List[ScoredItem]:
    """
    Stage 2: grade filtered items, with the local model for anything worth asking.

    An item the keyword stage rated above LLM_KEYWORD_THRESHOLD is sent to the
    model, which returns a graded 1-10 plus why/projects/category. Anything below
    the threshold, and anything past `max_llm_calls`, keeps its keyword-derived
    score — so the scale is continuous where a model looked and honest about
    where it did not.

    Args:
        items: List of FeedItem objects
        profile: Interest profile dictionary
        llm_call: Prompt->reply callable; defaults to the local model. Injectable
            so the call/No-call decision is testable without a running engine.
        max_llm_calls: Per-run budget; see LLM_MAX_CALLS.

    Sets ``stage2_score.last_llm_calls`` to the number of model calls made and
    ``stage2_score.last_graded`` to how many came back as a usable grade, so the
    run summary can tell "the model judged nothing" from "the model was not asked".

    Returns:
        List of ScoredItem objects
    """
    scorer = llm_call or call_local_llm
    use_model = llm_call is not None or os.environ.get("INTEL_DISABLE_LLM") != "1"

    scored_items = []
    all_projects = get_all_projects(profile)
    llm_calls = 0
    graded_calls = 0  # of those calls, how many produced a usable grade

    for item in items:
        # Combine title and summary for matching
        text = f"{item.title} {item.summary}"

        matched_topics = keyword_match(text, profile)
        kw_score = keyword_score(text, profile)
        fallback = _keyword_fallback(item, profile)

        relevance = fallback["relevance"]
        why = fallback["why"]
        matched_projects = match_projects(item, all_projects)
        category = determine_category(matched_topics)

        wants_model = (use_model and kw_score > LLM_KEYWORD_THRESHOLD
                       and llm_calls < max_llm_calls)
        if wants_model:
            llm_calls += 1
            try:
                graded = _parse_score_json(scorer(_score_prompt(item, profile)))
            except Exception as exc:  # transport/model failure -> keyword score
                graded = None
                print(f"  LLM scoring failed for {item.id}: {exc}")
            if graded:
                model_relevance = _clamp_relevance(graded.get("relevance"))
                if model_relevance is not None:
                    relevance = model_relevance
                    why = (graded.get("why") or "").strip() or why
                    projects = graded.get("projects")
                    if isinstance(projects, list) and projects:
                        matched_projects = [str(p) for p in projects]
                    model_category = (graded.get("category") or "").strip()
                    if model_category:
                        category = model_category
                graded_calls += 1

        scored_items.append(ScoredItem(
            id=item.id,
            source=item.source,
            title=item.title,
            url=item.url,
            summary=item.summary,
            discovered_at=item.discovered_at,
            authors=item.authors,
            source_tags=item.source_tags,
            relevance=relevance,
            urgency=determine_urgency(relevance),
            why=why,
            projects=matched_projects,
            category=category,
        ))

    # Both numbers, because they answer different questions: calls made says the
    # budget was spent, grades produced says the model contributed. Collapsed into
    # one, a model that answers with prose reads exactly like a model that scored
    # everything. Reported by run_scoring_pipeline.
    stage2_score.last_llm_calls = llm_calls
    stage2_score.last_graded = graded_calls
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
    profile: Optional[dict] = None,
    llm_call: Optional[Callable[[str], str]] = None,
) -> List[ScoredItem]:
    """
    Run the full two-stage scoring pipeline.
    
    Args:
        items: List of FeedItem objects
        profile: Interest profile (optional, loads default if not provided)
        llm_call: Optional stand-in for the local model (tests)
    
    Returns:
        List of ScoredItem objects
    """
    if profile is None:
        profile = load_profile()
    
    # Stage 1: Filter
    filtered = stage1_filter(items, profile)
    dropped = len(items) - len(filtered)
    if dropped:
        print(f"Stage 1 kept {len(filtered)} of {len(items)} items "
              f"({dropped} matched no interest keyword)")
    
    # Stage 2: Score
    scored = stage2_score(filtered, profile, llm_call=llm_call)

    # Calls and grades are different numbers, and the old line conflated them: an
    # engine that answered with prose made it say "LLM scored 1 of 1" while every
    # item still carried its keyword score. Read this line as "what the model
    # judged", and read a calls>grades gap as the engine misbehaving.
    calls = getattr(stage2_score, "last_llm_calls", 0)
    graded = getattr(stage2_score, "last_graded", 0)
    print(f"LLM scored {graded} of {calls} items it was asked about "
          f"(threshold {LLM_KEYWORD_THRESHOLD}); the rest kept their keyword score")
    if calls and not graded:
        print(f"  WARNING: the model at {LLM_URL} answered {calls} request(s) "
              "without a usable grade — every relevance below is keyword-only. "
              "Check the engine is up and still replying with JSON before "
              "trusting this run's scores.")

    return scored
