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
import re
import urllib.error
import urllib.request
from datetime import datetime
from typing import Callable, List, Optional, Dict, Any

from .models import (FeedItem, ScoredItem, GRADE_CALL_CAP, GRADE_KEYWORD,
                     GRADE_MODEL, GRADE_NO_USABLE_GRADE)
from ._paths import GRADE_STORE
from .profile import load_profile, keyword_match, keyword_score, get_all_projects
# The floor the second half of the #2092 warning is measured against. vault_writer
# imports models/profile/state/body/_paths and never scoring, so this direction adds
# no cycle; and the number is imported rather than typed because a warning quoting a
# floor that has since moved would be the same misreport this warning exists to
# prevent, one step further down the chain.
from .vault_writer import RELEVANCE_FLOOR

# Stage 2 only runs for items the keyword stage already rates above this.
# Matches the design the `intelligence-pipeline` skill documents ("LLM
# relevance scoring, only if keyword score > 0.3"); with the loader now reading
# real weights, 0.3 is a floor an unweighted topic (1.0) clears and a
# non-match (0.0) never does.
LLM_KEYWORD_THRESHOLD = 0.3

# ── a topic weighted at/below that threshold is off, and the run says so (#2092) ──
#
# `**Weight:**` in `interests.md` reads like a dial and behaves like a switch.
# `keyword_score` is `max(weight)` over the topics an item matched
# (profile.py:206-208) and stage 2 asks the model only above the threshold above
# (:336), so a topic at or below 0.3 gets no grade for an item that matches only
# it; what is left is `_keyword_fallback`'s `round(weight * 10)` (scoring.py:211),
# which at that weight cannot clear the vault writer's relevance floor of 4
# (vault_writer.py:47). The item is therefore neither graded nor written.
#
# Nothing between the profile and the vault said so. Stage 1 keeps the item on any
# keyword match whatever its weight (scoring.py:249), it leaves stage 2 carrying
# `grade_source=keyword` — the SAME value an engine outage leaves, so the cause is
# invisible — and the writer's only remark is a bare count: `Held N item(s) below
# relevance floor 4` (vault_writer.py:678), naming neither topic nor weight. So a
# person who sets 0.2 to "turn the interest down a little" sees their topic's items
# vanish from the vault with no sentence anywhere explaining that a weight is an
# off switch. This is a profile-level report, printed whether or not today's items
# matched the topic, because the topic that produces no symptom is the one whose
# silence is misleading: an item matching a 0.2 topic AND a 1.0 topic is graded and
# filed under the other topic (`max(matched, key=weight)`, vault_writer.py:277), so
# a low-weight topic loses its shared matches without ever showing a loss.
SWITCHED_OFF_MARK = "SWITCHED OFF:"


def topics_switched_off(profile: dict,
                        threshold: float = LLM_KEYWORD_THRESHOLD) -> List[Dict[str, Any]]:
    """Topics whose declared weight sits at or below the stage-2 call threshold.

    `<=`, not `<`: eligibility is `keyword_score > threshold` (scoring.py:336), so a
    topic at exactly the threshold gets no call either. A topic declaring no
    `**Weight:**` takes the loader's 1.0 default (profile.py:62) and never lands
    here, which is why the shipped `interests.md` — which declares none — prints
    nothing.
    """
    return [topic for topic in profile.get("topics", [])
            if float(topic.get("weight", 1.0)) <= threshold]


# The pipeline runs under a 1800 s task timeout (autonomy task #30) and a model
# call costs seconds. Uncapped, a bad day of items becomes a timeout, which is
# the failure mode the fleet already tracks for other tasks. Items past the cap
# keep their keyword-derived score and say so in the run output.
LLM_MAX_CALLS = 40
LLM_TIMEOUT_SECONDS = 90

# ── which survivor the call budget is spent on (backlog #2081) ────────────────
#
# Spending `LLM_MAX_CALLS` in the caller's list order made an ungraded item a
# positional property of two config files *and of the clock*. The raw day file is
# opened `"a"` by `state.save_raw_items` (state.py:77), so it accumulates every scan
# pass of the day, and each pass appends its own block: that pass's GitHub rows in
# `github-repos.yml` order, then its videos in feed order. A survivor's index
# therefore says which pass found it and where it sat inside that pass — nothing
# about the item. On the first day the cap bound — 2026-10-02, 42 survivors, four
# `discovered_at` minute-stamps at 05:02, 13:10, 21:12 and 21:13Z — the two items
# never asked were
# simply the last two rows of the file, both YouTube videos, each still
# carrying the keyword fallback's relevance 10. All 33 GitHub items got a call, and
# 20 of those 33 grades then fell below `vault_writer.RELEVANCE_FLOOR` and wrote
# nothing, against 1 of the 7 graded videos. More than half a capped budget bought
# grades the floor threw away while the writer refused the day's two headlining
# items.
#
# The key below is the cheapest stage-1 signal that actually discriminates, and it
# needs no model call to read. Components, highest priority first:
#
# 1. feed source, by the rank table. On that day 20 of 33 graded GitHub items fell
#    below the floor against 1 of 7 graded videos, so a call spent on a commit was
#    far likelier to be discarded than one spent on a video.
# 2. the stage-1 `keyword_score`, hardest match first — the same number that
#    decided the item was worth asking, so the cap bites the marginal admits. It is
#    second rather than first because today it carries no ordering information at
#    the top of the scale: it is `max(topic weight)` and all 42 survivors score
#    exactly 1.0, which is why ordering by it alone is a measured no-op (it refuses
#    the same two videos). Weights are Alan's call — ruled 2026-09-27 on #1380 — so
#    this key reads them rather than inventing replacements, and the day they are
#    re-scaled the budget follows them without a second change here.
# 3. the caller's position, last, purely to make the order total and stable.
#
# What this does NOT do: it never changes what an item past the cap may do. Those
# still carry `GRADE_CALL_CAP`, and the vault writer still refuses them (#1380).
SOURCE_ALLOCATION_RANK: Dict[str, int] = {"youtube": 0, "github": 1}

# A source with no measured overflow cost does not get to jump the feeds that have
# one, so an unranked source is allocated after every ranked one.
SOURCE_ALLOCATION_RANK_DEFAULT = 1 + max(SOURCE_ALLOCATION_RANK.values())


def source_allocation_rank(source: str) -> int:
    """Allocation rank of a feed source; lower is asked first. See the table above."""
    return SOURCE_ALLOCATION_RANK.get(source, SOURCE_ALLOCATION_RANK_DEFAULT)


def stage2_allocation_order(items: List[FeedItem], profile: dict) -> List[int]:
    """Indices of `items` in the order the stage-2 call budget is spent (#2081).

    Every index appears exactly once, so this is an ordering of the whole list, not
    a shortlist: items the keyword stage rated below `LLM_KEYWORD_THRESHOLD` are
    never candidates for a call wherever they land, and the caller's own order is
    preserved by `stage2_score` on the way out — the allocation decides who is
    asked, never the shape of the day file the writer reads back.
    """
    def rank(index: int):
        item = items[index]
        return (source_allocation_rank(item.source),
                -keyword_score(f"{item.title} {item.summary}", profile),
                index)

    return sorted(range(len(items)), key=rank)


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


# ── a keyword inside an address is not a keyword in the text (#2463) ────────────────
#
# `_keyword_occurs` bounds a single-word keyword with `\b` (profile.py:166-175), and `.`,
# `/` and `-` are all NON-word characters — so `ai` matches inside `figure.ai` and `gr00t`
# inside `…/gr00t-n1_6/` exactly as loudly as either would in a sentence. Measured on
# 2026-10-09 across the 17 retained raw day files (2,595 rows, 732 stage-1 keeps): 16 rows
# are kept ONLY by a keyword inside a URL domain or path segment, all 16 of them GitHub
# rows and 0 of the 1,316 YouTube ones. The false-positive shape #2241's owed check
# predicted, from a gate widened in #2241 to read the whole description.
#
# So both gate reads below match on this function's copy of the gate text with the URL
# substrings taken out. Match time and nowhere near: FeedItem.stage1_text(), the stored
# gate_description and summary, the intel-<date>.jsonl row and vault_writer's _entry_body
# render all still carry the addresses whole, which is why neither this module's writer nor
# body.py has a line of it — and why the scorer test
# test_url_ablation_is_match_time_only_stored_and_rendered_bytes_are_unchanged reads the
# day file back instead of trusting this comment.
#
# Two residuals, stated because a partial guard read as a whole one is the worse failure.
# `[^ ]+` spans a newline, so a URL that ends a line also takes the first token of the
# line after it; and a scheme-less host (tinyurl.com/…, the bare-host branch of
# _LINK_PRESENT_RE at body.py:202) is not a URL to this pattern, so a keyword inside one
# can still carry a keep. Both are recorded on #2463, along with what the ablation gives up:
# 16 of the corpus's 732 stage-1 keeps (all of them GitHub rows, 5% of its GitHub keeps),
# and 3 of the 33 rows the frozen corpus in tests/test_intel_pipeline_scorer.py admits under
# the whole-word rule. Whether the gate should instead carry the repo's own name would
# reverse the 2026-09-11 ruling quoted in `stage1_filter`'s docstring above — repo traffic
# is not an interest signal — so it is a separate decision owed on that item, not taken here.
_GATE_URL_RUN = re.compile(r"https?://[^ ]+|www[^ ]+")


def gate_match_text(item: FeedItem) -> str:
    """The gate text with every URL substring replaced by a single space.

    A space and not the empty string: deleting an address would splice the prose on either
    side of it into one token and could invent a match at the seam, where a space keeps the
    word boundaries the matcher reads where they were.

    This is the ONLY place the ablation exists, and it is a matcher input: nothing stored,
    rendered or published passes through it. See the block above for why (#2463).
    """
    return _GATE_URL_RUN.sub(" ", item.stage1_text())


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

    What it reads is `item.stage1_text()`, not `item.summary`, since #2241. Matching
    and publishing have different needs on one piece of channel copy: the scanner
    strips a description that is nothing but a link block so `knowledge/` never holds
    a link farm, and doing that before the gate meant such a video was matched on its
    title alone, with the profile keyword in the channel's own label never in front of
    the matcher. Stage 2's three other readers of `item.summary` (`_score_prompt`,
    `_keyword_fallback`, `match_projects`) still read the stripped value — widening
    them is an open scope question on that item, not a change to make here.

    Since #2463 the match runs on `gate_match_text(item)` rather than on `stage1_text()`
    itself: the same text with its URL substrings taken out, because `\b` cannot tell an
    address from a sentence and a keyword inside a domain was carrying keeps the profile
    never asked for. What is stored and published is untouched.
    """
    filtered_items = []

    for item in items:
        # Title plus the widest copy of the description we hold — the pre-strip text
        # where the scanner carried one, the stored summary otherwise — with the
        # addresses off it, for matching only (#2463, see `gate_match_text`).
        text = gate_match_text(item)

        # Check against profile keywords
        matched = keyword_match(text, profile)

        if matched:
            filtered_items.append(item)

    return filtered_items


# ── the first model grade an item id ever got, kept on disk (#2139) ──────────
#
# `temperature: 0` and `seed: 0` (#1232) do not pin the grade, because the flip is
# not sampling. Measured on 2026-10-03 by replaying byte-identical
# `_score_prompt` output for `youtube:UC0C-17n9iuUQPylguM1d-lQ:pvXmMntEIPY`
# against the live `primary` engine: relevance 3 five times in a row with nothing
# else in flight, then 4 once two other items' prompts were interleaved between the
# calls. The server runs `--enable-prefix-caching` with MTP speculation, fp8 KV and
# `--async-scheduling`, so the answer depends on batch state. `RELEVANCE_FLOOR = 4`
# with a strict `<` (vault_writer.py:47, :53) turns that 3-vs-4 into write-vs-refuse,
# and the two artifacts the day produces then disagree by construction:
# `--score` truncates and rewrites `intel-<date>.jsonl` wholesale
# (`__main__.py:150`) so it keeps the newest grade, while the writer dedupes by
# item id (`is_written`, vault_writer.py:229) so the vault keeps the first one that
# cleared the floor.
#
# The only layer that can pin an answer the engine will not pin is a store keyed by
# item id, so that is what this is: one JSON line per grade, appended, never
# rewritten, first row per id winning. Nothing about the sampler, the floor or the
# prompt moves.
#
# What it costs, stated plainly: a grade is a point-in-time judgement, so if Alan
# re-weights `interests.md` an item graded last month keeps the grade it got. That
# is the designed semantics — the alternative is the coin flip above — and the store
# is a plain JSONL a person can delete to re-grade the world. Its growth is one line
# per graded id (tens a day), bounded by the retention sweep like the other stores.
def load_grade_store() -> Dict[str, dict]:
    """Item id -> the first model grade it received, read back from the store.

    A row with no id, no parseable JSON, or a relevance `_clamp_relevance` refuses
    is skipped rather than fatal. The store grows for the life of the machine and a
    scoring pass must not die on one bad line in it — and a grade the store cannot
    read degrades to the pre-#2139 behaviour (ask the model again), never to a
    refused run.
    """
    grades: Dict[str, dict] = {}
    try:
        raw = GRADE_STORE.read_text()
    except OSError:
        return grades  # no store yet: a fresh machine, or a redirected test path
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        item_id = row.get("id")
        relevance = _clamp_relevance(row.get("relevance"))
        if not isinstance(item_id, str) or not item_id or relevance is None:
            continue
        if item_id in grades:
            continue  # append-only, first write wins: a later row cannot revise it
        why = row.get("why")
        category = row.get("category")
        projects = row.get("projects")
        grades[item_id] = {
            "relevance": relevance,
            "why": why if isinstance(why, str) else "",
            "projects": [str(p) for p in projects] if isinstance(projects, list) else [],
            "category": category if isinstance(category, str) else "",
        }
    return grades


def append_grade(item_id: str, relevance: int, why: str = "",
                 projects: Optional[List[str]] = None, category: str = "") -> None:
    """Append one item id's first model grade. Never rewrites the store.

    An unwritable store is said out loud and the grade still stands: the run must
    not lose a grade it already has because the cache could not be fed.
    """
    row = {"id": item_id, "relevance": relevance, "why": why,
           "projects": list(projects or []), "category": category}
    try:
        GRADE_STORE.parent.mkdir(parents=True, exist_ok=True)
        with open(GRADE_STORE, "a") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError as exc:
        print(f"  could not record the grade for {item_id} in {GRADE_STORE}: {exc}")


def _apply_model_grade(graded: dict, fallback_why: str,
                       fallback_projects: List[str], fallback_category: str):
    """(relevance, why, projects, category) from a grade, or None if it has none.

    One applier for both routes a grade can arrive by — the live reply from
    `_parse_score_json`, and a row read back from the store — because the defect
    #2139 is about is the day file and the vault disagreeing, and a cache that
    restored only the relevance would still rewrite `why`, `projects` and
    `category` on every pass.
    """
    relevance = _clamp_relevance(graded.get("relevance"))
    if relevance is None:
        return None
    why = (graded.get("why") or "").strip() or fallback_why
    projects = fallback_projects
    graded_projects = graded.get("projects")
    if isinstance(graded_projects, list) and graded_projects:
        projects = [str(p) for p in graded_projects]
    category = fallback_category
    graded_category = (graded.get("category") or "").strip()
    if graded_category:
        category = graded_category
    return relevance, why, projects, category


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

    Which eligible survivors the ``max_llm_calls`` calls are spent on is decided by
    ``stage2_allocation_order`` (#2081) — feed source first, then the stage-1
    keyword score — and not by where an item happened to sit in ``items``, which is
    the order the day's scan passes appended their rows and says nothing about the
    item. The list returned
    is still in the caller's order.

    An eligible item whose id already has a grade in the store
    (``load_grade_store``, #2139) is not put to the model again: it comes back with
    that grade and ``grade_source=model``, because the engine does not answer the
    same prompt twice once its batch state moves and ``RELEVANCE_FLOOR`` makes the
    difference a write/refuse decision. A stored grade costs no call, and it does not
    buy another item out of the budget either — it occupies the allocation slot it
    would have occupied, so the ids the cap bites are the same ids they were before
    the store existed and a second pass over one day's raw set reproduces that day
    instead of spending its freed budget on the items the first pass never reached.
    An id with no stored grade behaves exactly as it did before: asked live, in
    allocation order, until the budget is spent.

    Sets ``stage2_score.last_llm_calls`` to the number of model calls made
    (cache hits are not calls), ``stage2_score.last_graded`` to how many came back as
    a usable grade, ``stage2_score.last_cap_refused`` to how many eligible survivors
    the budget never reached, and ``stage2_score.last_grade_cache_hits`` to how many
    grades came from the store, so the run summary can tell "the model judged
    nothing" from "the model was not asked" from "the model ran out of calls" from
    "the model was asked about these on an earlier pass".

    Returns:
        List of ScoredItem objects
    """
    scorer = llm_call or call_local_llm
    use_model = llm_call is not None or os.environ.get("INTEL_DISABLE_LLM") != "1"

    scored_by_position: Dict[int, ScoredItem] = {}
    all_projects = get_all_projects(profile)
    # Re-read per pass, not memoised at import: the store is written by the scan
    # pass that graded the item and read by the next one, which is a different
    # process (`python -m intel_pipeline --score` runs once per scheduled tick).
    grades = load_grade_store()
    llm_calls = 0
    graded_calls = 0  # of those calls, how many produced a usable grade
    cap_refused = 0   # eligible survivors the budget never reached
    cache_hits = 0    # eligible survivors answered by the store instead of the model
    # Allocation slots spent — cache hits included, calls included. The cap is
    # compared against THIS and not against `llm_calls`, so a cached id cannot free
    # a call for an item the first pass never reached: which ids go ungraded stays a
    # property of `stage2_allocation_order` alone (#2081), exactly as it was before
    # any store existed, and that is what makes two passes agree item for item.
    budget_used = 0

    # Allocation order, caller order out (#2081) — see stage2_allocation_order.
    for index in stage2_allocation_order(items, profile):
        item = items[index]
        # The stripped copy: what the fallback scores, what the prompt shows, what the
        # writer publishes. Eligibility no longer reads it — see below.
        text = f"{item.title} {item.summary}"

        matched_topics = keyword_match(text, profile)
        # Whether to ASK is judged on the text stage 1 gated on, not on the stripped
        # summary (#2314). Since #2241 the gate admits an item on `stage1_text()` — title
        # plus the pre-strip description where a scanner carries one — so an item kept on
        # the channel's own label arrives here scoring 0.0 on `text`, which is not above
        # `LLM_KEYWORD_THRESHOLD`, so it is never a candidate for a call and ships
        # `_keyword_fallback`'s stand-in with no topic matched at all: relevance 1 on the
        # recorded instance (`youtube:UCLKPca3kwwd-B59HNr-_lvA:X6l4lpA0_NY`, 2026-10-06,
        # 2,148 gate chars matching `ai-llms` at weight 1.0 against 498 stored chars
        # scoring 0.0), the day's only `grade_source: keyword` row of 37, under
        # `RELEVANCE_FLOOR`. A survivor slot spent on a grade nobody took.
        #
        # What moves is only WHO IS ASKED. `keyword_score` stays `max(topic weight)`, so a
        # topic weighted at or below the threshold still buys no call whichever copy its
        # keyword sat in — the off switch #2092 documents stays a property of the profile.
        # What stays on the stripped `summary`: the prompt (`_score_prompt`), the fallback's
        # own score, project matching, the allocation ranking above, and everything the
        # writer publishes; the ruling on those three is owed on #2314, not taken here. A
        # row with no pre-strip copy — every GitHub row, and every YouTube row written before
        # #2241 — has `stage1_text() == text`, so for those the decision is identical to the
        # one this line made before the change.
        #
        # Since #2463 the copy it reads is `gate_match_text(item)` — `stage1_text()` with
        # its URL substrings taken out — and it is the SAME function stage 1 gates on, so
        # the two reads cannot disagree about one item. An item whose only keyword sat
        # inside an address was already dropped by stage 1; leaving this line on the
        # unabridged text would have it rated 0.9 here and handed a survivor slot to a grade
        # no writer would ever take. What the ablation reaches is still only WHO IS ASKED:
        # the prompt, the fallback's own score, project matching and every published copy
        # keep reading `text`, exactly as the paragraph above rules.
        kw_score = keyword_score(gate_match_text(item), profile)
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
        # The store is consulted exactly where a call would otherwise be made, so an
        # ineligible item (engine off, or rated at/below the threshold) keeps its
        # keyword score and its `grade_source=keyword` whether or not it was ever
        # graded — a disabled engine must keep looking like a disabled engine.
        stored = grades.get(item.id) if eligible else None
        wants_model = (eligible and stored is None
                       and budget_used < max_llm_calls)
        if stored is not None:
            # Graded on an earlier pass, so the number the engine would produce
            # today is not the point: this item's relevance is the grade it got,
            # whatever a re-draw would have said. `_apply_model_grade` cannot return
            # None here — `load_grade_store` already dropped every row whose
            # relevance `_clamp_relevance` refuses — so the tuple unpack is safe.
            relevance, why, matched_projects, category = _apply_model_grade(
                stored, why, matched_projects, category)
            grade_source = GRADE_MODEL
            budget_used += 1
            cache_hits += 1
        elif eligible and not wants_model:
            # Same eligibility test as `wants_model`, so the one thing that can
            # have failed here is the budget: this item's relevance is a stand-in
            # for a grade that was never taken, which is what the writer refuses.
            grade_source = GRADE_CALL_CAP
            cap_refused += 1
        if wants_model:
            grade_source = GRADE_NO_USABLE_GRADE
            llm_calls += 1
            budget_used += 1
            try:
                graded = _parse_score_json(scorer(_score_prompt(item, profile)))
            except Exception as exc:  # transport/model failure -> keyword score
                graded = None
                print(f"  LLM scoring failed for {item.id}: {exc}")
            if graded:
                applied = _apply_model_grade(graded, why, matched_projects, category)
                if applied is not None:
                    relevance, why, matched_projects, category = applied
                    grade_source = GRADE_MODEL
                    # Only a usable grade is recorded: an id the engine answered
                    # with junk stays ungraded in the store, so the next pass asks
                    # again instead of locking in a non-grade.
                    append_grade(item.id, relevance, why=why,
                                 projects=matched_projects, category=category)
                graded_calls += 1

        scored_by_position[index] = ScoredItem(
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
            # `intel-<date>.jsonl` — the only file the writer reads. `gate_description`
            # was exactly that on 2026-10-06: `ScoredItem.to_dict` writes the key and
            # `from_dict` reads it back (`models.py:111`, `:163`), yet all 37 rows of
            # that day's file carried `""` while their raw rows held up to 2,148 chars,
            # because the dataclass default is what `to_dict` had to write here. Anyone
            # then probing the scored file for the text the gate saw read an empty
            # string and concluded the description was empty — the false 0. The gate text
            # now reaches the day file, where a reader can see what the item was matched
            # on; it stays out of `summary`, which is what gets published (#2314).
            published=item.published,
            gate_description=item.gate_description,
        )

    # Three numbers, because they answer three questions: calls made says the
    # budget was spent, grades produced says the model contributed, and cap
    # refused says how much of the day the budget did not cover at all. Collapsed
    # into one, a model that answers with prose reads exactly like a model that
    # scored everything, and a 40-call budget spent on 258 survivors reads exactly
    # like a day with 40 items. Reported by run_scoring_pipeline.
    stage2_score.last_llm_calls = llm_calls
    stage2_score.last_graded = graded_calls
    stage2_score.last_cap_refused = cap_refused
    stage2_score.last_grade_cache_hits = cache_hits
    # Back to the caller's order: the allocation above decides who is asked, and the
    # day file the writer reads must not change shape because the budget moved.
    return [scored_by_position[i] for i in range(len(items))]


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
    cache_hits = getattr(stage2_score, "last_grade_cache_hits", 0)
    print(f"LLM scored {graded} of {calls} items it was asked about "
          f"(threshold {LLM_KEYWORD_THRESHOLD}); the rest kept their keyword score")
    # Said separately because the line above counts calls, and a pass that asked
    # nothing has to say why: on the second `--score` of one day those numbers are 0
    # calls and 0 grades, which reads exactly like an engine outage unless the pass
    # that reused the day's grades out of the store says where they came from.
    if cache_hits:
        print(f"  REUSED: {cache_hits} of {len(scored)} survivors were already "
              f"graded and were not put to the model again — their relevance is the "
              f"first grade their id ever received, from {GRADE_STORE}.")
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

    # A property of the loaded profile, not of the day's items: no `if filtered`, no
    # reference to `scored` below, because the topic this names is the one that
    # produced no visible symptom today (#2092 — see topics_switched_off).
    for topic in topics_switched_off(profile):
        weight = float(topic.get("weight", 1.0))
        keyword_relevance = max(1, min(10, int(round(weight * 10))))
        print(f"  {SWITCHED_OFF_MARK} topic {topic['name']!r} carries weight "
              f"{weight:g}, at or below the stage-2 keyword threshold "
              f"{LLM_KEYWORD_THRESHOLD:g}, so an item matching only that topic is "
              f"never put to the model — it can be neither graded nor written: its "
              f"keyword score {keyword_relevance} cannot clear the vault writer's "
              f"relevance floor {RELEVANCE_FLOOR}. A weight is an off switch, not a "
              f"dial; ranking an item is the model's grade, not its topic's weight.")

    return scored
