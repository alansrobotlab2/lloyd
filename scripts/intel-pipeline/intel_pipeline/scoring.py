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

from .models import (FeedItem, ScoredItem, GRADE_CALL_CAP, GRADE_KEYWORD,
                     GRADE_MODEL, GRADE_NO_USABLE_GRADE)
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

# Dropped titles the stage-1 line names before folding the rest into a count.
# A day's raw file holds ~25-100 rows, so an unbounded list would be most of
# the run log; twenty is enough to read a bad-recall day for what it is.
STAGE1_DROP_LIST_MAX = 20

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
        # Greedy and seeded (#1232): `vault_writer.RELEVANCE_FLOOR` turns this
        # integer into a keep/drop, so a sampled grade made an item's fate
        # depend on which run reached it first (one item scored 8, then 6).
        "temperature": 0,
        "seed": 0,
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
    the threshold keeps its keyword-derived score, and anything past
    `max_llm_calls` keeps that score too but is recorded as never graded, so the
    scale is continuous where a model looked and the run says out loud where it
    did not.

    Args:
        items: List of FeedItem objects
        profile: Interest profile dictionary
        llm_call: Prompt->reply callable; defaults to the local model. Injectable
            so the call/No-call decision is testable without a running engine.
        max_llm_calls: Per-run budget; see LLM_MAX_CALLS.

    Every item carries ``grade_source`` naming which of these produced its
    relevance: ``model`` (a usable grade), ``no_usable_grade`` (asked, nothing
    usable came back), ``call_cap`` (eligible, but the budget was already spent —
    so the model was never consulted about it), or ``keyword`` (never a candidate:
    the engine is off, or stage 1's score was at or below LLM_KEYWORD_THRESHOLD).
    Only ``model`` is a grade. The vault writer refuses ``call_cap`` and writes the
    other two, because the keyword fallback is the designed behaviour when the
    engine is unavailable and turning an outage into a zero-write day is not what
    the fallback is for.

    Sets ``stage2_score.last_llm_calls`` to the number of model calls made,
    ``stage2_score.last_graded`` to how many came back as a usable grade, and
    ``stage2_score.last_cap_refused`` to how many eligible survivors the budget
    never reached, so the run summary can tell "the model judged nothing" from "the
    model was not asked" from "the model ran out of calls".

    Returns:
        List of ScoredItem objects
    """
    scorer = llm_call or call_local_llm
    use_model = llm_call is not None or os.environ.get("INTEL_DISABLE_LLM") != "1"

    scored_items = []
    all_projects = get_all_projects(profile)
    llm_calls = 0
    graded_calls = 0  # of those calls, how many produced a usable grade
    cap_refused = 0   # eligible survivors the budget never reached

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
        # The fallback score until something replaces it, with the cause named:
        # an item is GRADE_KEYWORD here only when it was never a candidate for a
        # call at all (engine off, or rated at/below the threshold).
        grade_source = GRADE_KEYWORD

        eligible = use_model and kw_score > LLM_KEYWORD_THRESHOLD
        wants_model = eligible and llm_calls < max_llm_calls
        if eligible and not wants_model:
            # Same eligibility test as `wants_model`, so the one thing that can
            # have failed here is the budget: this item's relevance is a stand-in
            # for a grade that was never taken, which is what the writer refuses.
            grade_source = GRADE_CALL_CAP
            cap_refused += 1
        if wants_model:
            grade_source = GRADE_NO_USABLE_GRADE
            llm_calls += 1
            try:
                graded = _parse_score_json(scorer(_score_prompt(item, profile)))
            except Exception as exc:  # transport/model failure -> keyword score
                graded = None
                print(f"  LLM scoring failed for {item.id}: {exc}")
            if graded:
                model_relevance = _clamp_relevance(graded.get("relevance"))
                if model_relevance is not None:
                    grade_source = GRADE_MODEL
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
            grade_source=grade_source,
            # Second drop site (#1379): this rebuild is field-by-field, so a field
            # the scanner passes but this omits never reaches
            # `intel-<date>.jsonl` — the only file the writer reads.
            published=item.published,
        ))

    # Three numbers, because they answer three questions: calls made says the
    # budget was spent, grades produced says the model contributed, and cap
    # refused says how much of the day the budget did not cover at all. Collapsed
    # into one, a model that answers with prose reads exactly like a model that
    # scored everything, and a 40-call budget spent on 258 survivors reads exactly
    # like a day with 40 items. Reported by run_scoring_pipeline.
    stage2_score.last_llm_calls = llm_calls
    stage2_score.last_graded = graded_calls
    stage2_score.last_cap_refused = cap_refused
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
    max_llm_calls: Optional[int] = None,
) -> List[ScoredItem]:
    """
    Run the full two-stage scoring pipeline.
    
    Args:
        items: List of FeedItem objects
        profile: Interest profile (optional, loads default if not provided)
        llm_call: Optional stand-in for the local model (tests)
        max_llm_calls: stage-2 call budget, defaulting to LLM_MAX_CALLS. Passable
            because the summary's survivor-vs-cap gap is the thing under test and
            `stage2_score`'s own default binds at import time: without this, the
            only way to put a run over the cap is to wait for a real day that
            reaches it.
    
    Returns:
        List of ScoredItem objects
    """
    cap = LLM_MAX_CALLS if max_llm_calls is None else max_llm_calls
    if profile is None:
        profile = load_profile()
    
    # Stage 1: Filter
    filtered = stage1_filter(items, profile)
    kept_ids = {id(item) for item in filtered}
    dropped_items = [item for item in items if id(item) not in kept_ids]
    if dropped_items:
        print(f"Stage 1 kept {len(filtered)} of {len(items)} items "
              f"({len(dropped_items)} matched no interest keyword)")
        # By title, because a count cannot be read for recall: on 2026-09-15
        # the gate kept 2 of 27, three plainly on-interest videos went with the
        # 25, and the run log held nothing a reader could disagree with — a
        # bad-recall day and a quiet day printed the same line (#1155).
        for item in dropped_items[:STAGE1_DROP_LIST_MAX]:
            print(f"  dropped: {item.title}")
        more = len(dropped_items) - STAGE1_DROP_LIST_MAX
        if more > 0:
            print(f"  (+{more} more)")
    
    # Stage 2: Score
    scored = stage2_score(filtered, profile, llm_call=llm_call, max_llm_calls=cap)

    # Calls and grades are different numbers, and the old line conflated them: an
    # engine that answered with prose made it say "LLM scored 1 of 1" while every
    # item still carried its keyword score. Read this line as "what the model
    # judged", and read a calls>grades gap as the engine misbehaving.
    calls = getattr(stage2_score, "last_llm_calls", 0)
    graded = getattr(stage2_score, "last_graded", 0)
    cap_refused = getattr(stage2_score, "last_cap_refused", 0)
    print(f"LLM scored {graded} of {calls} items it was asked about "
          f"(threshold {LLM_KEYWORD_THRESHOLD}); the rest kept their keyword score")
    # Said separately, because the line above cannot carry it: `calls` is the
    # budget, not the survivor count, so on 2026-09-22 a fully-spent budget over
    # 258 survivors printed "LLM scored 40 of 40" — a sentence that reads as a
    # graded day while 218 items were never put to the model at all.
    if cap_refused:
        print(f"  NEVER ASKED: {cap_refused} of {len(scored)} survivors went "
              f"ungraded because the stage-2 call budget of {cap} was already "
              f"spent. Their relevance is an ungraded keyword score, and the vault "
              f"writer refuses them (grade_source={GRADE_CALL_CAP}).")
    if calls and not graded:
        print(f"  WARNING: the model at {LLM_URL} answered {calls} request(s) "
              "without a usable grade — every relevance below is keyword-only. "
              "Check the engine is up and still replying with JSON before "
              "trusting this run's scores.")

    return scored
