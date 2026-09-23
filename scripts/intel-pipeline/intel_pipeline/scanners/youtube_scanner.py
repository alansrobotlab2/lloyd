"""YouTube RSS feed scanner for monitoring channels."""

import json
import hashlib
import os
import re
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any, NamedTuple, Tuple
import time
import urllib.request
import xml.etree.ElementTree as ET

from ..models import FeedItem
from .. import state
from ..profile import load_profile


def _http_get(url: str, headers: Optional[Dict] = None, timeout: int = 30) -> str:
    """HTTP GET using stdlib urllib. Returns text content."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def load_youtube_channels_config() -> List[Dict[str, Any]]:
    """Load YouTube channels configuration.

    The previous hand-rolled line parser required 4-space indentation and
    never read the top-level ``channels:`` key, so it returned 0 of the 65
    configured channels while every run still logged success.
    """
    from pathlib import Path

    import yaml

    config_path = Path.home() / "lloyd/scripts/intel-pipeline/config/youtube-channels.yml"

    if not config_path.exists():
        return []

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    channels = config.get("channels", []) if isinstance(config, dict) else config
    return [c for c in channels if isinstance(c, dict) and c.get("channel_id")]


# State keys for YouTube scanner
YOUTUBE_STATE_KEY = "youtube_channels"

# Per-run feed coverage, written by every scan: how many of the feeds it tried
# to read it actually read. Without this the state file cannot distinguish
# "channels idle" from "feed endpoint unreachable", which is how a run that
# lost 55 of 64 feeds to 404/500 reported success (backlog #739).
COVERAGE_STATE_KEY = "youtube_coverage"

# The feed endpoint. Module-level and env-overridable, the same shape as
# `scoring.LLM_URL`, so a test can put a stub server on the wire instead of
# reaching youtube.com.
RSS_FEED_URL = os.environ.get(
    "INTEL_YOUTUBE_RSS_URL", "https://www.youtube.com/feeds/videos.xml")

# Attempts per channel, first try included. Upstream answers 404/500 for
# well-formed channel ids intermittently — 55/64 lost on 2026-09-10, 60/64 on
# 09-09, 37/64 on 09-12, 64/64 on 09-13 — and one shot is also indistinguishable
# from a quiet night. Measured on 12 channels during triage: 8/12 on a single
# attempt, 10/12 with up to 4 attempts at 5/10/20 s.
FETCH_ATTEMPTS = 4

# Wait before the 2nd attempt; each following wait doubles, so a dead channel
# costs 5 s + 10 s + 20 s = 35 s of sleeping plus four requests. The env override
# is operational, not a test hook: it shortens the ladder by hand when the
# endpoint is already known dead (the run of 2026-09-20 needed it to finish at
# all). What it does not do is bound the ladder — BACKOFF_BUDGET_SECONDS below
# does that, because unbounded it is 64 channels x 35 s = 2240 s of sleeping
# across config/youtube-channels.yml before a single request is counted, which
# alone is past the 1800 s `timeout_seconds` of autonomy task #30, the only job
# that runs this pipeline, and a run killed mid-stage writes nothing to the vault.
RETRY_BASE_WAIT_SECONDS = 5.0
RETRY_WAIT_ENV = "INTEL_YOUTUBE_RETRY_WAIT_SECONDS"

# Ceiling on sleeping inside ONE scan, drawn on by retry backoff and by the
# inter-channel pace alike. A backoff wait that no longer fits stops the scan
# backing off for good and every later channel is asked once, so the most any
# scan can spend asleep is 240 s whatever the channel count — where the unbounded
# ladder above would spend 2240 s. That bound is what keeps a full blackout
# inside task #30's 1800 s: 240 s of sleeping plus 64 single requests, instead of
# a run killed during this stage that reaches neither scoring nor the vault writer
# (backlog #1281, measured 2026-09-20 at ~50 s per dead channel).
# Arithmetic pinned by tests/test_intel_pipeline_scorer.py.
BACKOFF_BUDGET_SECONDS = 240.0
BACKOFF_BUDGET_ENV = "INTEL_YOUTUBE_BACKOFF_BUDGET_SECONDS"

# Rate-limit pause between channels, also counted against the budget above.
CHANNEL_PACING_SECONDS = 0.2

# Fetched-below-this-fraction-of-attempted is a degraded YouTube stage: more
# feeds were unreachable than reachable.
COVERAGE_FLOOR = 0.5


class FeedCoverage(NamedTuple):
    """How many feeds a scan was able to read, out of how many it tried.

    `attempted` counts only channels that got an outcome, so a channel skipped
    for having no id in the config is not silently in the denominator.
    """

    fetched: int
    attempted: int

    @property
    def degraded(self) -> bool:
        """True when more feeds were unreachable than reachable.

        An attempted count of 0 is not degradation: a scan with nothing to try
        says nothing about the endpoint, and "No YouTube channels configured"
        has already been printed in that case.
        """
        return self.attempted > 0 and self.fetched < self.attempted * COVERAGE_FLOOR

    def describe(self) -> str:
        return f"fetched {self.fetched} feeds of {self.attempted} attempted"


def sleep(seconds: float) -> None:
    """Sleep, at module level so a test can drive the retry ladder without wall clock."""
    time.sleep(seconds)


def _backoff_budget_seconds() -> float:
    """The scan's sleeping ceiling, with the operational env override applied.

    Read per scan rather than at import, like `_retry_wait_before` reads its own
    override, so a caller can set it for one run without editing the module.
    """
    override = os.environ.get(BACKOFF_BUDGET_ENV, "").strip()
    if override:
        return float(override)
    return BACKOFF_BUDGET_SECONDS


class SleepBudget:
    """Cumulative ceiling on sleeping inside one scan (backlog #1281).

    Every sleep the scan issues asks this first: retry backoff, and the
    inter-channel pace. That is what makes the bound hold regardless of how many
    channels are configured — 64 or 640, the scan cannot sleep past
    `budget_seconds`.

    A refused backoff is a decision, not just a skipped wait: it flips
    `backoff_exhausted`, which reduces that channel and every later one to a
    single attempt. Refusing per-channel instead would abandon a feed halfway up
    its ladder, and intermittent feeds are the normal case here (8/12 recovered on
    one attempt, 10/12 within four). So the ladder runs normally while the
    cumulative wait fits, and only the channel that would breach the bound is cut
    short. A refused pace only skips a pause; it never shortens anyone's ladder.
    """

    def __init__(self, budget_seconds: float):
        self.budget_seconds = float(budget_seconds)
        self.slept_seconds = 0.0
        self.backoff_exhausted = False
        self.channels_reduced = 0

    def allow_backoff(self, seconds: float) -> bool:
        """Grant `seconds` of retry sleeping, or refuse it and stop retrying.

        Firing is sticky: once the budget is spent the answer is no for the rest of
        the scan, so the channels after it are asked once each — and only those are
        counted, since the channel that spent the last of the budget still got the
        attempts that fit inside it.
        """
        if self.backoff_exhausted:
            self.channels_reduced += 1
            return False
        if self._grant(seconds):
            return True
        self.backoff_exhausted = True
        return False

    def allow_pace(self) -> bool:
        """Grant one inter-channel pause, or skip it once the budget is spent."""
        return self._grant(CHANNEL_PACING_SECONDS)

    def describe_reduction(self) -> str:
        """The line the scan prints when the budget fired, naming the reduction.

        A reader of the run record has to be able to tell "we stopped asking" from
        "the endpoint is dead", and the coverage line alone cannot say it: the
        denominator still holds every channel.
        """
        return (f"  Backoff budget of {self.budget_seconds:g} s spent "
                f"({self.slept_seconds:.1f} s slept): stopped backing off and tried "
                f"{self.channels_reduced} remaining channel(s) once each")

    def _grant(self, seconds: float) -> bool:
        if self.slept_seconds + seconds > self.budget_seconds:
            return False
        self.slept_seconds += seconds
        return True


def _retry_wait_before(attempt: int) -> float:
    """Seconds to wait before `attempt` (2 or later): 5 s, then 10 s, then 20 s."""
    base = RETRY_BASE_WAIT_SECONDS
    override = os.environ.get(RETRY_WAIT_ENV, "").strip()
    if override:
        base = float(override)
    return base * (2 ** (attempt - 2))


def _parse_feed_entries(content: str) -> List[Dict]:
    """Parse an Atom feed body into the video dicts the scanner consumes."""
    root = ET.fromstring(content)

    # Define namespaces
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "media": "http://search.yahoo.com/mrss/",
        "yt": "http://www.youtube.com/xml/schemas/2015"
    }

    entries = root.findall("atom:entry", namespaces)
    videos = []

    for entry in entries:
        # Extract fields
        entry_id = entry.find("atom:id", namespaces)
        title = entry.find("atom:title", namespaces)
        published = entry.find("atom:published", namespaces)
        summary = entry.find("atom:summary", namespaces)
        media_desc = entry.find("media:description", namespaces)
        link = entry.find("atom:link", namespaces)

        if entry_id is None or title is None:
            continue

        video_id = entry_id.text.split(":")[-1] if entry_id.text else ""
        video_url = link.get("href", "") if link is not None else ""
        description = media_desc.text if media_desc is not None else (summary.text if summary is not None else "")

        videos.append({
            "id": video_id,
            "title": title.text.strip() if title.text else "",
            "url": video_url,
            "description": description.strip() if description else "",
            "published": published.text if published is not None else ""
        })

    return videos


def fetch_channel_rss(channel_id: str,
                      budget: Optional[SleepBudget] = None) -> Tuple[List[Dict], bool]:
    """Fetch the RSS feed of one channel, retrying transient upstream failures.

    Returns ``(videos, fetched)``. ``fetched`` is False only when every attempt
    failed — the feed was unreachable — so an empty list with ``fetched=True``
    means an idle channel. Merging those two is the defect this exists for:
    a run that could not reach the endpoint is otherwise indistinguishable from
    a night with nothing new.

    A body that arrives but does not parse counts as a failed attempt too (an
    empty or truncated 200 is not a feed we read), and so is retried.

    Pass ``budget`` (the per-scan ``SleepBudget``, backlog #1281) to bound the
    sleeping this fetch may add to the scan: a backoff wait that no longer fits
    ends the ladder here and marks the budget exhausted, so later channels are
    asked once. With no budget the ladder is the full ``FETCH_ATTEMPTS``, which is
    what a caller outside a scan gets.
    """
    rss_url = f"{RSS_FEED_URL}?channel_id={channel_id}"
    last_error: Optional[BaseException] = None
    attempts_made = 0

    for attempt in range(1, FETCH_ATTEMPTS + 1):
        if attempt > 1:
            wait = _retry_wait_before(attempt)
            if budget is not None and not budget.allow_backoff(wait):
                break
            sleep(wait)
        attempts_made = attempt
        try:
            content = _http_get(
                rss_url,
                headers={"User-Agent": "lloyd-intel-pipeline"},
                timeout=30
            )
            return _parse_feed_entries(content), True
        except Exception as e:
            last_error = e

    print(f"  Error fetching RSS for channel {channel_id} after {attempts_made} "
          f"of {FETCH_ATTEMPTS} attempts: {type(last_error).__name__}: {last_error}")
    return [], False


def scan_youtube_channels() -> Tuple[List[FeedItem], FeedCoverage]:
    """
    Scan configured YouTube channels for new videos.

    Returns:
        (items, coverage) — the new items and how many feeds the scan was able
        to read. The coverage is also persisted under `COVERAGE_STATE_KEY`, so
        the run report and the state file can both say whether the endpoint was
        reachable (backlog #739).
    """
    channels = load_youtube_channels_config()
    if not channels:
        print("No YouTube channels configured")
        return [], FeedCoverage(fetched=0, attempted=0)
    
    # Load current state
    current_state = state.load_state()
    if YOUTUBE_STATE_KEY not in current_state:
        current_state[YOUTUBE_STATE_KEY] = {}
    channel_state = current_state[YOUTUBE_STATE_KEY]
    
    all_items = []
    today = datetime.utcnow().strftime("%Y-%m-%d")
    feeds_fetched = 0
    feeds_attempted = 0
    # One sleeping ceiling for the whole scan (backlog #1281): the retry ladder is
    # bounded by cumulative waiting, not per channel, so a blackout costs at most
    # the budget and the run still reaches scoring and the vault writer inside the
    # 1800 s of autonomy task #30. Every channel is still tried and still counted,
    # so the coverage signal below fires on a blackout exactly as it did before.
    budget = SleepBudget(_backoff_budget_seconds())

    print(f"\nScanning {len(channels)} YouTube channels...")
    
    for i, channel in enumerate(channels):
        handle = channel.get("handle", "")
        name = channel.get("name", "")
        channel_id = channel.get("channel_id", "")
        
        if not channel_id:
            continue
        
        # Rate limiting between channels, paid from the same budget the backoff
        # spends: past the budget the pause is skipped rather than the channel.
        if i > 0 and budget.allow_pace():
            sleep(CHANNEL_PACING_SECONDS)

        print(f"  [{i+1}/{len(channels)}] {name} ({handle})...")

        # Get stored state for this channel
        stored_last_video = channel_state.get(channel_id, "")

        # Fetch RSS feed under the scan's sleeping budget. Each channel the
        # budget refuses to back off for is counted there, and the count is
        # reported once, after the loop, where the total is known.
        videos, fetched = fetch_channel_rss(channel_id, budget=budget)

        # Counted before the empty-feed branch below: that branch is where
        # "unreachable" and "idle" used to become the same thing, and a counter
        # placed after it could not see the difference it exists to record.
        feeds_attempted += 1
        if fetched:
            feeds_fetched += 1
        
        if not videos:
            continue
        
        # Process videos (newest first)
        found_new = False
        for video in videos:
            video_id = video.get("id", "")
            item_id = f"youtube:{channel_id}:{video_id}"
            
            # Check if we've reached already-seen videos
            if stored_last_video and video_id == stored_last_video:
                found_new = True
                break
            
            # Skip if already seen
            if state.is_seen(item_id, current_state):
                continue
            
            # Skip if we've hit the last known video (for efficiency)
            if stored_last_video and found_new:
                break
            
            # Create FeedItem
            title = video.get("title", "")
            url = video.get("url", "")
            description = video.get("description", "")[:500] if video.get("description") else ""
            published = video.get("published", "")
            
            item = FeedItem(
                id=item_id,
                source="youtube",
                title=title,
                url=url,
                summary=description,
                discovered_at=datetime.utcnow().isoformat() + "Z",
                authors=[name] if name else [],
                source_tags=[handle] if handle else [],
                # The value `_parse_feed_entries` already read (line 232) and the
                # local above already bound (line 390): passing it is the whole of
                # the fix, and it is what lets the writer hold a 453-day-old video
                # instead of filing it under today's heading (backlog #1379).
                published=published,
            )
            all_items.append(item)
            state.mark_seen(item_id, current_state)
        
        # Update state with most recent video ID
        if videos:
            channel_state[channel_id] = videos[0].get("id", "")
    
    # Say out loud when the bound was what ended the retrying, and how many
    # channels it ended up asking only once. Coverage counts below are unaffected:
    # every channel the budget shortened is still in the denominator, so a
    # blackout still reads as a blackout (backlog #1281).
    if budget.channels_reduced:
        print(budget.describe_reduction())

    # Save updated state, including this run's feed coverage
    coverage = FeedCoverage(fetched=feeds_fetched, attempted=feeds_attempted)
    current_state[YOUTUBE_STATE_KEY] = channel_state
    current_state[COVERAGE_STATE_KEY] = {"fetched": coverage.fetched,
                                         "attempted": coverage.attempted}
    state.save_state(current_state)

    # Save raw items
    if all_items:
        state.save_raw_items(all_items, today)
        print(f"\nSaved {len(all_items)} items to raw JSONL")
    
    return all_items, coverage


if __name__ == "__main__":
    # Example usage
    items, coverage = scan_youtube_channels()
    print(f"\n=== YouTube Scan Complete ===")
    print(f"Found {len(items)} new items")
    print(f"YouTube feed coverage: {coverage.describe()}")

    if coverage.degraded:
        print(f"YouTube stage degraded: {coverage.describe()}")
        sys.exit(1)

    for item in items[:5]:
        print(f"\n[{item.source}] {item.title}")
        print(f"  URL: {item.url}")
        print(f"  Authors: {item.authors}")
        print(f"  Tags: {item.source_tags}")
