#!/usr/bin/env python3
"""
AI Engineer YouTube Monitor
===========================
Autonomous pipeline that checks the @aiDotEngineer YouTube channel for new videos,
fetches transcripts, and creates structured knowledge notes in the vault.

Produces vault notes matching the knowledge base format:
- Rich frontmatter (segment, tags, type, domain, sources, summary)
- Executive Summary
- Key Facts / Technical Details
- Referenced entities (GitHub repos, papers, tools)
- Open Questions
- Sources

Usage:
    python3 ai-engineer-monitor.py --process-one    # Process next unprocessed video
    python3 ai-engineer-monitor.py --dry-run        # Show what's new
    python3 ai-engineer-monitor.py --backfill       # Register all channel videos
    python3 ai-engineer-monitor.py --list            # List all state
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request
from xml.etree import ElementTree as ET

# ── Configuration ──────────────────────────────────────────────────────

CHANNEL_ID = "UCLKPca3kwwd-B59HNr-_lvA"
CHANNEL_HANDLE = "@aiDotEngineer"
UPLOADS_PLAYLIST = f"https://www.youtube.com/playlist?list=UU{CHANNEL_ID[2:]}"
STATE_DIR = os.path.expanduser("~/.local/share/ai-engineer")
STATE_FILE = os.path.join(STATE_DIR, "seen.json")

VAULT_YT_DIR = os.path.expanduser("~/obsidian/knowledge/youtube/AI_Engineer")
VAULT_GH_DIR = os.path.expanduser("~/obsidian/knowledge/github")
VAULT_PAPER_DIR = os.path.expanduser("~/obsidian/knowledge/papers")
TMP_CLONES = os.path.expanduser("~/.cache/ai-engineer-clones")

MAX_FAILURE_RETRIES = 12
TRANSIENT_RETRIES = 3  # Transient failures (network, quota) before hard-fail
RETRY_INTERVAL_SECONDS = 900

# LLM endpoints
LLM_URL = os.environ.get("LLM_API_URL", "http://localhost:8096/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "primary")


def is_video_playable(video_id):
    """Quick check if a video is playable (not a scheduled premiere, not age-restricted, etc).
    Returns (playable: bool, reason: str)."""
    code = '''
import sys, json, re
try:
    import yt_dlp
except ImportError:
    print("NOT_PLAYABLE: yt-dlp not available")
    sys.exit(0)

ydl = yt_dlp.YoutubeDL({
    "quiet": True, "no_warnings": True, "extract_flat": False,
})
try:
    info = ydl.extract_info(sys.argv[1], download=False)
    status = info.get("status", "")
    availability = info.get("availability", "")
    reason = info.get("reason", "")
    # Check the raw playability status
    play_status = info.get("_playability_status", {})
    play_reason = play_status.get("reason", "") if play_status else ""
    print(json.dumps({
        "playable": status != "premiere_scheduled" and availability != "private" and availability != "unlisted",
        "status": status,
        "availability": availability,
        "reason": reason,
        "play_reason": play_reason,
    }))
except Exception as e:
    err_msg = str(e).lower()
    if "premiere" in err_msg or "unplayable" in err_msg or "not available" in err_msg:
        print("NOT_PLAYABLE: " + str(e)[:200])
    else:
        print("UNKNOWN: " + str(e)[:200])
'''
    result = subprocess.run(
        ["uv", "run", "--with", "yt-dlp", "python3", "-c", code,
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return False, "yt-dlp unavailable"
    output = result.stdout.strip()
    if output.startswith("NOT_PLAYABLE") or output.startswith("UNKNOWN"):
        return False, output.split(": ", 1)[-1] if ": " in output else output
    if output.startswith("{"):
        try:
            data = json.loads(output)
            if not data.get("playable"):
                return False, data.get("reason") or data.get("play_reason") or data.get("status", "unplayable")
            return True, ""
        except json.JSONDecodeError:
            pass
    return True, ""  # Assume playable if we can't determine


# ── State management ───────────────────────────────────────────────────

def ensure_dirs():
    for d in [STATE_DIR, VAULT_YT_DIR, VAULT_GH_DIR, VAULT_PAPER_DIR, TMP_CLONES]:
        os.makedirs(d, exist_ok=True)


def load_state():
    ensure_dirs()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"seen": {}, "backfill_complete_through": None}


def save_state(state):
    ensure_dirs()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── YouTube channel fetching ──────────────────────────────────────────

def fetch_channel_videos():
    """Fetch channel videos via yt-dlp (handles lazy-loaded channels)."""
    code = f'''
import sys, json
import yt_dlp

ydl_opts = {{
    'extract_flat': True,
    'ignoreerrors': True,
    'playlistend': 200,
    'quiet': True,
    'no_warnings': True,
}}
ydl = yt_dlp.YoutubeDL(ydl_opts)
info = ydl.extract_info('{UPLOADS_PLAYLIST}', download=False)
videos = []
if info and 'entries' in info:
    for entry in info['entries']:
        videos.append({{
            'id': entry.get('id', ''),
            'title': entry.get('title', ''),
            'upload_date': entry.get('upload_date', ''),
        }})
print(json.dumps(videos))
'''
    result = subprocess.run(
        ["uv", "run", "--with", "yt-dlp", "python3", "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"ERROR: yt-dlp fetch failed: {result.stderr[:500]}")
        return []

    try:
        videos = json.loads(result.stdout)
        # Sort by upload_date descending (newest first)
        videos.sort(key=lambda v: v.get("upload_date", ""), reverse=True)
        for v in videos:
            v["published"] = v.get("upload_date", "")
            v["url"] = f"https://www.youtube.com/watch?v={v['id']}"
        return videos
    except json.JSONDecodeError:
        print("ERROR: Failed to parse yt-dlp output")
        return []


# ── Transcript fetching ────────────────────────────────────────────────

def fetch_transcript(video_id):
    """Fetch transcript using youtube-transcript-api via uv."""
    code = '''
import sys
from youtube_transcript_api import YouTubeTranscriptApi
api = YouTubeTranscriptApi()
t = api.fetch(sys.argv[1], languages=["en", "en-US"])
segments = list(t)
print(" ".join(s.text for s in segments))
'''
    result = subprocess.run(
        ["uv", "run", "--with", "youtube-transcript-api", "python3", "-c", code, video_id],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  Transcript fetch failed: {result.stderr[:500]}")
        return None
    return result.stdout.strip()


# ── Video metadata via yt-dlp ─────────────────────────────────────────

def fetch_video_metadata(video_id):
    """Fetch detailed metadata via yt-dlp."""
    code = '''
import sys, json
import yt_dlp
ydl = yt_dlp.YoutubeDL({
    "quiet": True, "no_warnings": True, "extract_flat": False,
})
info = ydl.extract_info(sys.argv[1], download=False)
print(json.dumps({
    "title": info.get("title", ""),
    "upload_date": info.get("upload_date", ""),
    "channel": info.get("channel", ""),
    "description": info.get("description", "")[:3000],
}))
'''
    result = subprocess.run(
        ["uv", "run", "--with", "yt-dlp", "python3", "-c", code,
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


# ── LLM calls ──────────────────────────────────────────────────────────

def call_llm(system_prompt, user_content, max_tokens=2000):
    """Call the LLM. Handles both content and reasoning-only models."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    body = json.dumps(payload).encode()
    req = request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        resp = request.urlopen(req, timeout=180)
        result = json.loads(resp.read())
        msg = result["choices"][0]["message"]

        content = msg.get("content")
        if not content:
            reasoning = msg.get("reasoning", "")
            if reasoning:
                final_match = re.search(r"Final(?:[_ ]Answer)?[^\n]*\n\n(.*)", reasoning, re.DOTALL)
                if final_match:
                    content = final_match.group(1).strip()
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', reasoning.strip())
                    content = " ".join(sentences[-5:]) if len(sentences) > 1 else reasoning.strip()
        return content if content else None
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None


# ── Entity extraction ─────────────────────────────────────────────────

def extract_entities(transcript):
    """Extract GitHub URLs, arXiv IDs, paper URLs, tools, and named entities."""
    entities = {
        "github_urls": [], "paper_arxiv": [], "paper_urls": [],
        "other_urls": [], "tools": [], "named_entities": []
    }

    # GitHub URLs
    gh = re.findall(r"(https?://github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+)", transcript)
    entities["github_urls"] = list(dict.fromkeys(gh))

    # arXiv IDs
    arxiv_urls = re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", transcript)
    arxiv_bare = re.findall(r"\b(\d{4}\.\d{4,5})\b", transcript)
    entities["paper_arxiv"] = list(dict.fromkeys(arxiv_urls + arxiv_bare))

    # Paper URLs
    papers = re.findall(
        r"(https?://(openreview|paperswithcode|huggingface|neurips|icml\.cc|arxiv)\.[\w/_.-]+)",
        transcript, re.IGNORECASE
    )
    entities["paper_urls"] = list(dict.fromkeys(papers))

    # General URLs
    all_urls = set(re.findall(r"(https?://[\w\-.]+\.[\w\-.]+(?:/[\w\-.%#]+(?:\?[\w%&=.-]*)?)?)", transcript))
    excluded = set(entities["github_urls"]) | set(entities["paper_urls"])
    entities["other_urls"] = list(all_urls - excluded)[:20]

    return entities


# ── GitHub cloning & note creation ─────────────────────────────────────

def clone_and_note(github_url):
    """Clone a GitHub repo and create a vault note."""
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", github_url)
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    repo_slug = f"{owner}-{repo}"
    clone_path = os.path.join(TMP_CLONES, repo_slug)
    note_path = os.path.join(VAULT_GH_DIR, f"{repo_slug}.md")

    if os.path.exists(note_path):
        print(f"  GitHub note already exists: {note_path}")
        return {"owner": owner, "repo": repo, "note_path": note_path}

    print(f"  Cloning {owner}/{repo}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", github_url, clone_path],
            capture_output=True, timeout=120, check=True,
        )
    except Exception as e:
        print(f"  Clone failed: {e}")
        return None

    # Read README
    readme = ""
    for candidate in ["README.md", "README.MD", "README.rst", "README.txt", "README"]:
        rp = os.path.join(clone_path, candidate)
        if os.path.exists(rp):
            with open(rp) as f:
                readme = f.read(15000)
            break

    # Summarize via LLM
    summary = call_llm(
        "You are a technical analyst. Summarize this GitHub repository in 3-5 sentences: name, purpose, key features, tech stack.",
        f"Repository: {owner}/{repo}\n\nREADME:\n{readme[:8000]}",
        max_tokens=600,
    )

    # Count files
    file_count = 0
    for root_dir, dirs, files in os.walk(clone_path):
        dirs[:] = [d for d in dirs if d != ".git"]
        file_count += len(files)

    # Create vault note (knowledge format)
    now = datetime.now(timezone.utc).isoformat()
    note_content = f"""---
segment: knowledge
tags: [software, github]
type: reference
domain: software
source: {github_url}
repo: {owner}/{repo}
cloned_at: {now}
---

# {owner}/{repo}

## Summary
{summary if summary else "No summary available."}

## README
```
{readme[:5000]}
```

## Details
- **Files:** {file_count}
- **URL:** {github_url}
"""
    with open(note_path, "w") as f:
        f.write(note_content)
    print(f"  Created: {note_path}")

    # Cleanup
    subprocess.run(["rm", "-rf", clone_path], capture_output=True)
    return {"owner": owner, "repo": repo, "note_path": note_path}


# ── Paper fetching ─────────────────────────────────────────────────────

def fetch_arxiv_paper(arxiv_id):
    """Fetch paper info from arXiv API and create vault note."""
    match = re.search(r"(\d{4}\.\d{4,5})", arxiv_id)
    if not match:
        return None
    arxiv_id_clean = match.group(1)
    note_path = os.path.join(VAULT_PAPER_DIR, f"arxiv-{arxiv_id_clean}.md")

    if os.path.exists(note_path):
        print(f"  Paper note already exists: {note_path}")
        return {"arxiv_id": arxiv_id_clean, "note_path": note_path}

    try:
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id_clean}"
        req = request.Request(api_url, headers={"User-Agent": "Lloyd/1.0"})
        resp = request.urlopen(req, timeout=30)
        xml = resp.read().decode("utf-8")

        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(xml)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        authors = [a.text for a in entry.findall("atom:author/atom:name", ns)]
        published_el = entry.find("atom:published", ns)

        title = (title_el.text or "Unknown").strip().replace("\n", " ")
        summary = (summary_el.text or "").strip()
        published = (published_el.text or "").strip() if published_el is not None else ""
        categories = [c.get("term", "") for c in entry.findall("atom:category", ns)]

        note_content = f"""---
segment: knowledge
tags: [research, paper]
type: reference
domain: ai-research
source: https://arxiv.org/abs/{arxiv_id_clean}
arxiv_id: {arxiv_id_clean}
published: {published}
authors: {", ".join(authors[:10])}
---

# {title}

## Authors
{', '.join(authors[:20])}

## Categories
{', '.join(categories)}

## Abstract
{summary[:3000]}

## Links
- [Abstract](https://arxiv.org/abs/{arxiv_id_clean})
- [PDF](https://arxiv.org/pdf/{arxiv_id_clean}.pdf)
- [Source](https://arxiv.org/src/{arxiv_id_clean}/a)
"""
        with open(note_path, "w") as f:
            f.write(note_content)
        print(f"  Created: {note_path}")
        return {"arxiv_id": arxiv_id_clean, "title": title, "note_path": note_path}
    except Exception as e:
        print(f"  arXiv fetch failed: {e}")
        return None


def fetch_generic_paper(url):
    """Fetch paper info from generic paper URL."""
    slug_match = re.search(r"/([\w-]+(?:/[A-Za-z0-9_-]+)*)$", url.rstrip("/"))
    slug = slug_match.group(1) if slug_match else "paper"
    slug = re.sub(r"[^\w-]+", "-", slug.lower().replace(" ", "-"))[:100]
    note_path = os.path.join(VAULT_PAPER_DIR, f"{slug}.md")

    if os.path.exists(note_path):
        print(f"  Paper note already exists: {note_path}")
        return {"url": url, "note_path": note_path}

    try:
        req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = request.urlopen(req, timeout=30)
        html = resp.read().decode("utf-8", errors="replace")[:20000]

        title_match = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Unknown"

        desc_match = re.search(
            r'<meta\s+(?:name|property)="(?:description|og:description)"\s+content="([^"]+)"',
            html, re.IGNORECASE
        )
        description = desc_match.group(1) if desc_match else ""

        title_safe = re.sub(r"[^\w\s-]", "", title)[:100]
        title_slug = re.sub(r"[^\w-]+", "-", title_safe.lower().replace(" ", "-").strip())[:100] or slug
        final_path = os.path.join(VAULT_PAPER_DIR, f"{title_slug}.md")

        now = datetime.now(timezone.utc).isoformat()
        note_content = f"""---
segment: knowledge
tags: [research, external]
type: reference
source: {url}
fetched_at: {now}
---

# {title}

## Description
{description or "N/A"}

## Source
[{url}]({url})
"""
        with open(final_path, "w") as f:
            f.write(note_content)
        print(f"  Created: {final_path}")
        return {"url": url, "note_path": final_path}
    except Exception as e:
        print(f"  Generic paper fetch failed: {e}")
        return None


# ── Knowledge note generation ──────────────────────────────────────────

def generate_knowledge_note(video_id, title, publish_date, description, transcript, entities):
    """
    Use the LLM to generate a structured knowledge note from the transcript,
    matching the vault's deep-research format.
    """
    # Extract entities for the LLM prompt
    entity_context = ""
    if entities["github_urls"]:
        entity_context += f"\n\nGitHub References:\n" + "\n".join(f"- {u}" for u in entities["github_urls"])
    if entities["paper_arxiv"]:
        entity_context += f"\n\nArXiv References:\n" + "\n".join(f"- {a}" for a in entities["paper_arxiv"])
    if entities["paper_urls"]:
        entity_context += f"\n\nPaper URLs:\n" + "\n".join(f"- {u}" for u in entities["paper_urls"])

    prompt = f"""You are creating a structured knowledge note for a YouTube video.
Generate a complete vault knowledge note in the following format.

INPUT VIDEO:
- Title: {title}
- Channel: @aiDotEngineer (AI Engineer)
- Video ID: {video_id}
- Published: {publish_date}
- URL: https://www.youtube.com/watch?v={video_id}

DESCRIPTION:
{description[:2000]}

{entity_context}

TRANSCRIPT (abbreviated):
{transcript[:12000]}

OUTPUT FORMAT (use EXACTLY this structure, filling in the content):

---
segment: knowledge
tags: [youtube, ai, technology]
type: video-note
domain: ai
source: https://www.youtube.com/watch?v={video_id}
video_id: {video_id}
channel: aiDotEngineer
published: {publish_date}
---

# {title}

## Executive Summary

[2-4 sentence dense summary of what the video covers, the core argument, and why it matters. No filler.]

## Key Points

- [Key point 1]
- [Key point 2]
- [Key point 3]
- [Key point 4]

## Technical Details

[Detailed technical analysis of the main topic(s) covered. Specific claims, architectures, tools, methods. This is the meat of the note.]

## Tools & Frameworks Mentioned

- [tool/framework]: [brief description of how it was discussed]
- [tool/framework]: [brief description]

## Related Resources

### GitHub
- [GitHub links with wiki-style [[wiki links]]]
- [or "None detected" if no GitHub refs]

### Papers
- [Paper links with wiki-style [[wiki links]]]
- [or "None detected" if no paper refs]

### Links
- [Other relevant URLs]

## Open Questions

- [Unresolved questions raised by the video content]
- [Things that could be investigated further]

"""
    result = call_llm(
        "You are a knowledge engineer creating structured technical notes. Output ONLY the markdown note in the format shown — no preamble, no explanations, no conversational filler. Include frontmatter, executive summary, key points, technical details, tools, related resources, and open questions.",
        prompt,
        max_tokens=4000,
    )
    return result if result else None


# ── Video processing ──────────────────────────────────────────────────

def slugify(text, max_len=80):
    slug = re.sub(r"[^\w\s-]", "", text).lower().replace(" ", "-")[:max_len]
    return re.sub(r"-+", "-", slug).strip("-")


def process_video(video_id, title, published_text):
    """Process a single video end-to-end."""
    print(f"\n=== Processing: {title} ({video_id}) ===")

    # Fetch full metadata
    metadata = fetch_video_metadata(video_id)
    if metadata:
        title = metadata.get("title", title)
        publish_date = metadata.get("upload_date", "")[:10]
        description = metadata.get("description", "")
    else:
        publish_date = ""
        description = ""

    # Fetch transcript
    print("  Fetching transcript...")
    transcript = fetch_transcript(video_id)
    if not transcript:
        print("  ERROR: Could not fetch transcript")
        return False

    print(f"  Transcript: {len(transcript)} chars, {len(transcript.split())} words")

    # Extract entities
    print("  Extracting entities...")
    entities = extract_entities(transcript)
    for k, v in entities.items():
        if v:
            print(f"    {k}: {v}")

    # Process GitHub references
    github_results = []
    for gh_url in entities["github_urls"]:
        print(f"  Processing GitHub: {gh_url}")
        result = clone_and_note(gh_url)
        if result:
            github_results.append(result)

    # Process paper references
    paper_results = []
    for arxiv_id in entities["paper_arxiv"]:
        print(f"  Processing arXiv: {arxiv_id}")
        result = fetch_arxiv_paper(arxiv_id)
        if result:
            paper_results.append(result)

    for paper_url in entities["paper_urls"]:
        print(f"  Processing paper: {paper_url}")
        result = fetch_generic_paper(paper_url)
        if result:
            paper_results.append(result)

    # Generate knowledge note via LLM
    print("  Generating knowledge note...")
    knowledge_note = generate_knowledge_note(
        video_id, title, publish_date, description, transcript, entities
    )

    if not knowledge_note:
        # Fallback: basic structured note
        knowledge_note = f"""---
segment: knowledge
tags: [youtube, ai]
type: video-note
domain: ai
source: https://www.youtube.com/watch?v={video_id}
video_id: {video_id}
channel: aiDotEngineer
published: {publish_date}
---

# {title}

## Summary
Processing failed — transcript below.

## Transcript
{transcript[:8000]}
"""

    # Write note
    video_slug = slugify(title)
    date_str = publish_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    note_filename = f"{date_str}-{video_slug}.md"
    note_path = os.path.join(VAULT_YT_DIR, note_filename)

    with open(note_path, "w") as f:
        f.write(knowledge_note)
    print(f"  Created vault note: {note_path}")
    return True


# ── Main ────────────────────────────────────────────────────────────────

def _is_retry_eligible(entry, now):
    """A failed entry is retry-eligible if it hasn't exceeded MAX_FAILURE_RETRIES
    and at least RETRY_INTERVAL_SECONDS has passed since the last attempt.

    WHY: transcripts often appear 1–2 hours after a video is published; a
    transcript-fetch failure on the first try is normal and should be retried.

    However, *permanent* errors (no transcript exists, video unavailable, age
    restricted) should be hard-failed after TRANSIENT_RETRIES attempts to avoid
    blocking the queue indefinitely."""
    if entry.get("status") != "failed":
        return False
    failure_count = entry.get("failure_count", 0)
    error = entry.get("transcript_error", "")

    # Permanent errors: give up after TRANSIENT_RETRIES
    if _is_permanent_error(error):
        if failure_count >= TRANSIENT_RETRIES:
            return False
        retries_used = failure_count
    else:
        retries_used = failure_count

    if retries_used >= MAX_FAILURE_RETRIES:
        return False
    last_attempt = entry.get("last_attempt_at")
    if not last_attempt:
        return True
    try:
        last_dt = datetime.fromisoformat(last_attempt)
        return (now - last_dt).total_seconds() >= RETRY_INTERVAL_SECONDS
    except (ValueError, TypeError):
        return True


def _is_permanent_error(error_msg):
    """Return True if the transcript error is permanent (not a network glitch)."""
    if not error_msg:
        return False
    low = error_msg.lower()
    permanent_keywords = [
        "transcriptdata", "notranscriptfound", "no subtitles",
        "no transcript", "age restriction", "unavailable", "private",
        "deleted", "premiere", "unplayable", "region",
        "videounplayable",
        "couldn't find", "could not fetch transcript", "could not find transcript",
    ]
    return any(kw in low for kw in permanent_keywords)


def _is_transient_unplayable(reason):
    """Return True if a video is unplayable for a reason that resolves on its own —
    a scheduled premiere or an upcoming/in-progress live stream that becomes a
    normal, transcribable video once it airs. These must NOT be permanently
    skipped; they're deferred and re-checked each run until they go live."""
    if not reason:
        return False
    low = reason.lower()
    transient_keywords = [
        "premiere", "live event", "will begin", "begins in",
        "starts in", "upcoming", "scheduled", "live stream will",
        "this live stream",
    ]
    return any(kw in low for kw in transient_keywords)


def mark_video_skipped(state, video_id, reason):
    """Mark a video as permanently skipped (unavailable, private, deleted,
    region-locked, age-restricted). For premieres/livestreams that haven't aired
    yet, use mark_video_deferred() instead — those become playable later."""
    if "seen" not in state:
        state["seen"] = {}
    now = datetime.now(timezone.utc).isoformat()
    state["seen"][video_id] = {
        "status": "skipped",
        "reason": reason,
        "skipped_at": now,
    }


def mark_video_deferred(state, video_id, reason, video=None):
    """Mark a video as temporarily unavailable (premiere/livestream not yet aired).
    Unlike skipped, deferred entries are re-checked every run via is_video_playable
    and processed once they go live — so a talk first seen as a premiere is never
    dropped permanently."""
    if "seen" not in state:
        state["seen"] = {}
    now = datetime.now(timezone.utc).isoformat()
    prior = state["seen"].get(video_id)
    entry = prior if isinstance(prior, dict) else {}
    entry.update({"status": "deferred", "reason": reason, "deferred_at": now})
    if video:
        entry.setdefault("title", video.get("title", ""))
        entry.setdefault("published", video.get("published", ""))
    state["seen"][video_id] = entry


def get_next_video(state, all_videos):
    """Find next video to process.

    Priority:
      1. Unseen videos (truly new, newest first per channel ordering)
         — skip unplayable ones (premieres, region-locked, etc.)
      2. Failed videos eligible for retry — never let one failed video at the
         top of the feed block fresh uploads behind it."""
    seen = state["seen"]
    now = datetime.now(timezone.utc)
    for video in all_videos:
        vid_id = video["id"]
        entry = seen.get(vid_id)
        # Unseen, or previously deferred (transiently unplayable) → (re)check
        # playability. A premiere/livestream stays deferred until it airs.
        if entry is None or (isinstance(entry, dict) and entry.get("status") == "deferred"):
            playable, reason = is_video_playable(vid_id)
            if not playable:
                if _is_transient_unplayable(reason):
                    print(f"  Deferring not-yet-available video {vid_id}: {reason}")
                    mark_video_deferred(state, vid_id, reason, video)
                else:
                    print(f"  Skipping unplayable video {vid_id}: {reason}")
                    mark_video_skipped(state, vid_id, reason)
                save_state(state)
                continue
            return video
    for video in all_videos:
        entry = seen.get(video["id"])
        if isinstance(entry, dict) and _is_retry_eligible(entry, now):
            return video
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Engineer YouTube Monitor")
    parser.add_argument("--dry-run", action="store_true", help="Show new videos without processing")
    parser.add_argument("--backfill", action="store_true", help="Register all channel videos for backfill")
    parser.add_argument("--process-one", action="store_true", help="Process next unprocessed video")
    parser.add_argument("--list", action="store_true", help="List state")
    args = parser.parse_args()

    ensure_dirs()
    state = load_state()

    # Fetch channel videos
    print("Fetching channel videos...")
    all_videos = fetch_channel_videos()
    print(f"Found {len(all_videos)} videos on channel")

    # List mode
    if args.list:
        print(f"\nState has {len(state['seen'])} seen videos:")
        for vid, info in state["seen"].items():
            print(f"  {vid}: {info['title']} — {info['status']}")
        return

    # Dry run
    if args.dry_run:
        seen_ids = set(state["seen"].keys())
        for v in all_videos:
            status = "NEW" if v["id"] not in seen_ids else "seen"
            print(f"  [{status}] {v.get('published', '?')} {v['title']} ({v['id']})")
        return

    # Backfill mode
    if args.backfill:
        print("Backfill: registering all channel videos for processing")
        for v in all_videos:
            if v["id"] not in state["seen"]:
                state["seen"][v["id"]] = {
                    "title": v["title"],
                    "published": v.get("published", ""),
                    "status": "pending",
                }
        save_state(state)
        pending = [k for k, v in state["seen"].items() if v["status"] == "pending"]
        print(f"  {len(pending)} pending videos registered")
        return

    # Process one (default)
    # Force-process any entry not handled by another path. Statuses with their
    # own handling: completed/skipped (terminal), failed (retry path below),
    # deferred (playability re-check in get_next_video). Everything else —
    # including orphaned/legacy statuses like "new" that older script versions
    # wrote but the current loop never consumed — is reprocessed here so it
    # never silently stalls.
    _NOT_PENDING = {"completed", "skipped", "failed", "deferred"}
    pending_vids = [
        (vid, info) for vid, info in state["seen"].items()
        if isinstance(info, dict) and info.get("status") not in _NOT_PENDING
    ]

    # Also check for retry-eligible failed videos
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    retry_candidates = []
    for vid, info in state["seen"].items():
        if isinstance(info, dict) and info.get("status") == "failed" and _is_retry_eligible(info, now):
            retry_candidates.append((vid, info))

    if retry_candidates:
        vid, info = retry_candidates[0]
        title = info.get("title", "Unknown")
        print(f"Retrying failed video: {title} ({vid})")
        success = process_video(vid, title, info.get("published", ""))
        prior_count = info.get("failure_count", 0)
        state["seen"][vid]["status"] = "completed" if success else "failed"
        if not success:
            state["seen"][vid]["failure_count"] = prior_count + 1
            state["seen"][vid]["last_attempt_at"] = now.isoformat()
        else:
            state["seen"][vid].pop("failure_count", None)
            state["seen"][vid].pop("last_attempt_at", None)
        state["seen"][vid]["youtube_note"] = os.path.join(
            VAULT_YT_DIR, f"video-{vid}.md"
        )
        save_state(state)
        if success:
            print(f"\n✓ Retry succeeded: {title}")
        else:
            print(f"\n✗ Retry failed: {title}")
        return

    if pending_vids:
        vid, info = pending_vids[0]
        title = info.get("title", "Unknown")
        print(f"Processing backfill video: {title} ({vid})")
        success = process_video(vid, title, info.get("published", ""))
        state["seen"][vid]["status"] = "completed" if success else "failed"
        state["seen"][vid]["youtube_note"] = os.path.join(
            VAULT_YT_DIR, f"video-{vid}.md"
        )
        save_state(state)
        if success:
            print(f"\n✓ Processed: {title}")
        else:
            print(f"\n✗ Failed: {title}")
        return

    # Otherwise check for new videos
    next_video = get_next_video(state, all_videos)
    if next_video:
        success = process_video(
            next_video["id"],
            next_video["title"],
            next_video.get("published", ""),
        )
        note_path = os.path.join(VAULT_YT_DIR, f"video-{next_video['id']}.md")
        if success:
            state["seen"][next_video["id"]] = {
                "title": next_video["title"],
                "published": next_video.get("published", ""),
                "status": "completed",
                "youtube_note": note_path,
            }
            save_state(state)
            print(f"\n✓ Processed: {next_video['title']}")
        else:
            prior = state["seen"].get(next_video["id"], {})
            failure_count = prior.get("failure_count", 0) + 1
            state["seen"][next_video["id"]] = {
                "title": next_video["title"],
                "published": next_video.get("published", ""),
                "status": "failed",
                "youtube_note": note_path,
                "failure_count": failure_count,
                "last_attempt_at": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state)
            if failure_count >= MAX_FAILURE_RETRIES:
                print(f"\n✗ Failed (giving up after {failure_count} attempts): {next_video['title']}")
            else:
                print(f"\n✗ Failed (attempt {failure_count}/{MAX_FAILURE_RETRIES}, will retry): {next_video['title']}")
    else:
        print("No new videos to process. Channel caught up.")


if __name__ == "__main__":
    main()
