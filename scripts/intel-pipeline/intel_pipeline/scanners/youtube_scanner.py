"""YouTube RSS feed scanner for monitoring channels."""

import requests
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
import time
import xml.etree.ElementTree as ET

from ..models import FeedItem
from .. import state
from ..profile import load_profile


# State keys for YouTube scanner
YOUTUBE_STATE_KEY = "youtube_channels"


def load_youtube_channels_config() -> List[Dict[str, Any]]:
    """Load YouTube channels configuration."""
    import yaml
    from pathlib import Path
    
    config_path = Path.home() / "lloyd/scripts/intel-pipeline/config/youtube-channels.yml"
    
    if not config_path.exists():
        return []
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config.get("channels", [])


def fetch_channel_rss(channel_id: str) -> List[Dict]:
    """Fetch RSS feed for a YouTube channel."""
    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    
    try:
        response = requests.get(
            rss_url,
            headers={"User-Agent": "lloyd-intel-pipeline"},
            timeout=30
        )
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
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
    
    except requests.RequestException as e:
        print(f"  Error fetching RSS for channel {channel_id}: {e}")
        return []
    except ET.ParseError as e:
        print(f"  Error parsing RSS for channel {channel_id}: {e}")
        return []


def scan_youtube_channels() -> List[FeedItem]:
    """
    Scan configured YouTube channels for new videos.
    
    Returns:
        List of FeedItem objects
    """
    channels = load_youtube_channels_config()
    if not channels:
        print("No YouTube channels configured")
        return []
    
    # Load current state
    current_state = state.load_state()
    if YOUTUBE_STATE_KEY not in current_state:
        current_state[YOUTUBE_STATE_KEY] = {}
    channel_state = current_state[YOUTUBE_STATE_KEY]
    
    all_items = []
    today = datetime.utcnow().strftime("%Y-%m-%d")
    
    print(f"\nScanning {len(channels)} YouTube channels...")
    
    for i, channel in enumerate(channels):
        handle = channel.get("handle", "")
        name = channel.get("name", "")
        channel_id = channel.get("channel_id", "")
        
        if not channel_id:
            continue
        
        # Rate limiting: 0.2s between requests
        if i > 0:
            time.sleep(0.2)
        
        print(f"  [{i+1}/{len(channels)}] {name} ({handle})...")
        
        # Get stored state for this channel
        stored_last_video = channel_state.get(channel_id, "")
        
        # Fetch RSS feed
        videos = fetch_channel_rss(channel_id)
        
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
    
    # Save updated state
    current_state[YOUTUBE_STATE_KEY] = channel_state
    state.save_state(current_state)
    
    # Save raw items
    if all_items:
        state.save_raw_items(all_items, today)
        print(f"\nSaved {len(all_items)} items to raw JSONL")
    
    return all_items


if __name__ == "__main__":
    # Example usage
    items = scan_youtube_channels()
    print(f"\n=== YouTube Scan Complete ===")
    print(f"Found {len(items)} new items")
    
    for item in items[:5]:
        print(f"\n[{item.source}] {item.title}")
        print(f"  URL: {item.url}")
        print(f"  Authors: {item.authors}")
        print(f"  Tags: {item.source_tags}")
