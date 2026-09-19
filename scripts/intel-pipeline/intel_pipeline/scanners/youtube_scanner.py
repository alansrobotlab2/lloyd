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

# Wait before the 2nd attempt; each following wait doubles, so a channel gets
# 5 s, 10 s, 20 s — 35 s of waiting per dead feed. The env override is
# operational, not a test hook: a blackout across all 64 configured channels
# would cost 64 × 35 s ≈ 37 minutes of retrying, past the 1800 s
# `timeout_seconds` on autonomy task #30, which runs this pipeline.
RETRY_BASE_WAIT_SECONDS = 5.0
RETRY_WAIT_ENV = "INTEL_YOUTUBE_RETRY_WAIT_SECONDS"

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


def fetch_channel_rss(channel_id: str) -> Tuple[List[Dict], bool]:
    """Fetch the RSS feed of one channel, retrying transient upstream failures.

    Returns ``(videos, fetched)``. ``fetched`` is False only when every attempt
    failed — the feed was unreachable — so an empty list with ``fetched=True``
    means an idle channel. Merging those two is the defect this exists for:
    a run that could not reach the endpoint is otherwise indistinguishable from
    a night with nothing new.

    A body that arrives but does not parse counts as a failed attempt too (an
    empty or truncated 200 is not a feed we read), and so is retried.
    """
    rss_url = f"{RSS_FEED_URL}?channel_id={channel_id}"
    last_error: Optional[BaseException] = None

    for attempt in range(1, FETCH_ATTEMPTS + 1):
        if attempt > 1:
            sleep(_retry_wait_before(attempt))
        try:
            content = _http_get(
                rss_url,
                headers={"User-Agent": "lloyd-intel-pipeline"},
                timeout=30
            )
            return _parse_feed_entries(content), True
        except Exception as e:
            last_error = e

    print(f"  Error fetching RSS for channel {channel_id} after {FETCH_ATTEMPTS} "
          f"attempts: {type(last_error).__name__}: {last_error}")
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
    
    print(f"\nScanning {len(channels)} YouTube channels...")
    
    for i, channel in enumerate(channels):
        handle = channel.get("handle", "")
        name = channel.get("name", "")
        channel_id = channel.get("channel_id", "")
        
        if not channel_id:
            continue
        
        # Rate limiting: 0.2s between requests
        if i > 0:
            sleep(0.2)
        
        print(f"  [{i+1}/{len(channels)}] {name} ({handle})...")
        
        # Get stored state for this channel
        stored_last_video = channel_state.get(channel_id, "")
        
        # Fetch RSS feed
        videos, fetched = fetch_channel_rss(channel_id)

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
                source_tags=[handle] if handle else []
            )
            all_items.append(item)
            state.mark_seen(item_id, current_state)
        
        # Update state with most recent video ID
        if videos:
            channel_state[channel_id] = videos[0].get("id", "")
    
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
