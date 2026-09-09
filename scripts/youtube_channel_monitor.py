#!/usr/bin/env python3
"""
YouTube Channel Monitor
=======================
Autonomous pipeline that checks one tracked YouTube channel for new videos,
fetches transcripts, and creates structured knowledge notes in the vault.

Channels live in the CHANNELS registry below; pick one with --channel. Each
channel has its own state file and vault directory, and every other stage
(transcripts, GitHub/arXiv enrichment, note generation) is shared. This file
began life as `ai-engineer-monitor.py`; that name is now a shim that runs
this with `--channel ai-engineer`, so autonomy task #75 did not change.

Two ways to turn a video into a note:

* **Session path (the default in production).** The `youtube-digest` worker
  source asks this script for the next video (`--register-new`, `--pending`),
  has it fetch the transcript and metadata into a bundle directory
  (`--fetch`), and then runs a real session — Inner Voice on, transcript in
  the session list — that reads the bundle, writes the note, judges whether
  the video holds anything that would improve Lloyd, and files a draft
  backlog item when it does. The session reports back through `--complete`
  / `--fail`. This script never calls the model on that path.
* **Script path (`--process-one`, `--process-all`).** The original
  everything-in-one-process digest: transcript → direct LLM call → note. Kept
  for operator use and as the fallback; nothing visible comes out of it but
  the note.

Produces vault notes matching the knowledge base format:
- Rich frontmatter (segment, tags, type, domain, sources, summary)
- Executive Summary
- Key Facts / Technical Details
- Referenced entities (GitHub repos, papers, tools)
- Open Questions
- Sources

Usage:
    python3 youtube_channel_monitor.py --channel discover-ai --process-one   # next unprocessed video
    python3 youtube_channel_monitor.py --channel discover-ai --dry-run       # show what's new
    python3 youtube_channel_monitor.py --channel discover-ai --since-days 60 # register the last 60 days
    python3 youtube_channel_monitor.py --channel discover-ai --process-all   # drain pending (operator)
    python3 youtube_channel_monitor.py --channel discover-ai --backfill      # register every channel video
    python3 youtube_channel_monitor.py --channel discover-ai --list          # list state
    python3 youtube_channel_monitor.py --channel discover-ai --pending --json    # what the worker may take
    python3 youtube_channel_monitor.py --channel discover-ai --fetch <id> --json # bundle for one video
    python3 youtube_channel_monitor.py --channel discover-ai --eval-report       # regenerate the report note

`--since-days` also records a *floor*: the oldest video inside the window.
The new-video walk stops there, so a channel tracked from a date does not get
crawled back through its whole history one tick at a time (which is exactly
what a channel registered with plain `--backfill` is asking for).
"""

import json
import os
import re
import tempfile
import subprocess
from datetime import datetime, timezone
from urllib import request
from xml.etree import ElementTree as ET

# ── Configuration ──────────────────────────────────────────────────────

# One entry per tracked channel. `handle` is written into note front matter
# as `channel:` (no @) — existing AI Engineer notes carry `aiDotEngineer`, so
# that value must not change. `state_dir` and `vault_dir` are per channel;
# GitHub/paper notes and the clone cache are shared.
CHANNELS = {
    "ai-engineer": {
        "channel_id": "UCLKPca3kwwd-B59HNr-_lvA",
        "handle": "aiDotEngineer",
        "name": "AI Engineer",
        "state_dir": "~/.local/share/ai-engineer",
        "vault_dir": "~/obsidian/knowledge/youtube/AI_Engineer",
    },
    "discover-ai": {
        "channel_id": "UCfOvNb3xj28SNqPQ_JIbumg",
        "handle": "code4AI",
        "name": "Discover AI",
        "state_dir": "~/.local/share/discover-ai",
        "vault_dir": "~/obsidian/knowledge/youtube/Discover_AI",
    },
}
DEFAULT_CHANNEL = "ai-engineer"

# Module globals the pipeline reads. They default to the AI Engineer channel
# so importing this module behaves exactly like the old script; configure()
# repoints them and must run before anything touches state or the vault.
CHANNEL_KEY = DEFAULT_CHANNEL
CHANNEL_ID = CHANNELS[DEFAULT_CHANNEL]["channel_id"]
CHANNEL_HANDLE = CHANNELS[DEFAULT_CHANNEL]["handle"]
CHANNEL_NAME = CHANNELS[DEFAULT_CHANNEL]["name"]
UPLOADS_PLAYLIST = f"https://www.youtube.com/playlist?list=UU{CHANNEL_ID[2:]}"
STATE_DIR = os.path.expanduser(CHANNELS[DEFAULT_CHANNEL]["state_dir"])
STATE_FILE = os.path.join(STATE_DIR, "seen.json")
VAULT_YT_DIR = os.path.expanduser(CHANNELS[DEFAULT_CHANNEL]["vault_dir"])


def configure(channel_key):
    """Point the module at one channel from CHANNELS.

    The pipeline functions read module globals at call time, so rebinding
    them here is enough — it keeps the diff against the single-channel
    script small and leaves every stage's behaviour unchanged.
    """
    global CHANNEL_KEY, CHANNEL_ID, CHANNEL_HANDLE, CHANNEL_NAME, UPLOADS_PLAYLIST
    global STATE_DIR, STATE_FILE, VAULT_YT_DIR
    if channel_key not in CHANNELS:
        raise KeyError(f"unknown channel {channel_key!r}; known: {', '.join(sorted(CHANNELS))}")
    ch = CHANNELS[channel_key]
    CHANNEL_KEY = channel_key
    CHANNEL_ID = ch["channel_id"]
    CHANNEL_HANDLE = ch["handle"]
    CHANNEL_NAME = ch["name"]
    UPLOADS_PLAYLIST = f"https://www.youtube.com/playlist?list=UU{CHANNEL_ID[2:]}"
    STATE_DIR = os.path.expanduser(ch["state_dir"])
    STATE_FILE = os.path.join(STATE_DIR, "seen.json")
    VAULT_YT_DIR = os.path.expanduser(ch["vault_dir"])
    _NOTE_INDEX.clear()
    return ch


def channel_info():
    """The active channel as a dict — what lloyd_improvement_eval is handed."""
    return {
        "key": CHANNEL_KEY, "channel_id": CHANNEL_ID, "handle": CHANNEL_HANDLE,
        "name": CHANNEL_NAME, "state_dir": STATE_DIR, "vault_dir": VAULT_YT_DIR,
    }


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

    Returns (playable: bool, reason: str, upload_date: str).

    The date is a by-product, not a second probe: this is a full
    (`extract_flat: False`) extraction, so it already holds the one field the
    flat playlist listing does not carry. `register_new` used to drop it and
    record `published: ""`, and `pending_entries` sorts newest-first with ""
    last — so every genuinely new upload went to the *back* of a queue that
    exists to reach new uploads first. Invisible while a backlog is small;
    on 2026-09-09 AI Engineer had 170 pending and a video published that day
    would have waited behind all of them. An unresolvable date is still "",
    which is exactly the old behaviour.
    """
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
        "upload_date": info.get("upload_date") or "",
    }))
except Exception as e:
    err_msg = str(e).lower()
    if "premiere" in err_msg or "unplayable" in err_msg or "not available" in err_msg:
        print("NOT_PLAYABLE: " + str(e)[:200])
    else:
        print("UNKNOWN: " + str(e)[:200])
'''
    uv_path = os.path.expanduser("~/.local/bin/uv")
    result = subprocess.run(
        [uv_path, "run", "--with", "yt-dlp", "python3", "-c", code,
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return False, "yt-dlp unavailable", ""
    output = result.stdout.strip()
    if output.startswith("NOT_PLAYABLE") or output.startswith("UNKNOWN"):
        return False, output.split(": ", 1)[-1] if ": " in output else output, ""
    if output.startswith("{"):
        try:
            data = json.loads(output)
            date = data.get("upload_date") or ""
            if not data.get("playable"):
                return False, data.get("reason") or data.get("play_reason") or data.get("status", "unplayable"), date
            return True, "", date
        except json.JSONDecodeError:
            pass
    return True, "", ""  # Assume playable if we can't determine


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

def fetch_channel_videos(limit=200):
    """Fetch channel videos via yt-dlp (handles lazy-loaded channels).

    `limit` is the playlist cap. 200 covers a tick; a 60-day window on a
    conference channel needs more (AI Engineer posted ~280 in the 60 days to
    2026-09-08), so `--since-days` asks for 600.
    """
    code = f'''
import sys, json
import yt_dlp

ydl_opts = {{
    'extract_flat': True,
    'ignoreerrors': True,
    'playlistend': {int(limit)},
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
    uv_path = os.path.expanduser("~/.local/bin/uv")
    result = subprocess.run(
        [uv_path, "run", "--with", "yt-dlp", "python3", "-c", code],
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
    """Fetch transcript using youtube-transcript-api via uv, with yt-dlp fallback.

    Returns (transcript_text, error_msg). On success transcript_text is
    the text and error_msg is None. On failure transcript_text is None
    and error_msg contains the reason.
    """
    # ── Primary: youtube-transcript-api ────────────────────────────
    uv_path = os.path.expanduser("~/.local/bin/uv")
    code = '''
import sys
from youtube_transcript_api import YouTubeTranscriptApi
api = YouTubeTranscriptApi()
t = api.fetch(sys.argv[1], languages=["en", "en-US"])
segments = list(t)
print(" ".join(s.text for s in segments))
'''
    result = subprocess.run(
        [uv_path, "run", "--with", "youtube-transcript-api", "python3", "-c", code, video_id],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return result.stdout.strip(), None

    err_primary = result.stderr[:500].strip()
    print(f"  Primary transcript fetch failed: {err_primary}")

    # ── Fallback chain: yt-dlp VTT → HLS (m3u8) captions → failure ──

    def parse_vtt_to_text(raw_vtt):
        """Parse a VTT file into cleaned text."""
        clean = re.sub(r'<[^>]*>', ' ', raw_vtt)
        clean = re.sub(r'X-TIMESTAMP-MAP[^ ]*|Kind: \w+', '', clean)
        clean = re.sub(r'^WEBVTT\s*$', '', clean, flags=re.MULTILINE)
        lines = clean.split('\n')
        text_parts = []
        skip = False
        for line in lines:
            if '-->' in line:
                skip = True
                continue
            if skip:
                skip = False
                continue
            stripped = line.strip()
            if stripped:
                text_parts.append(stripped)
        # Deduplicate consecutive repeated lines
        deduped = []
        prev = None
        for p in text_parts:
            if p != prev:
                deduped.append(p)
                prev = p
        return " ".join(deduped)

    # 1a. Try yt-dlp --write-auto-sub (works for regular auto-captions)
    import glob
    with tempfile.NamedTemporaryFile(prefix='yt_sub_', suffix='.vtt', delete=False) as tmp:
        tmp_path = tmp.name
    base = tmp_path.replace('.vtt', '')
    dl_result = subprocess.run(
        ["yt-dlp", "--write-auto-sub", "--sub-lang", "en",
         "--skip-download", "--no-warnings",
         "-o", base, f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=60,
    )
    vtt_files = glob.glob(base + '*')
    vtt_file = None
    for f in vtt_files:
        if f.endswith('.en.vtt') and os.path.getsize(f) > 100:
            vtt_file = f
            break
    if vtt_file:
        with open(vtt_file) as fh:
            transcript = parse_vtt_to_text(fh.read())
        for f in vtt_files:
            try:
                os.unlink(f)
            except OSError:
                pass
        if len(transcript) > 200:
            print(f"  yt-dlp VTT fallback succeeded: {len(transcript)} chars")
            return transcript, None

    # 1b. Try HLS (m3u8) caption stream — used for live premiere captions
    # yt-dlp may not be able to download these directly
    try:
        info_result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", "--no-warnings",
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=30,
        )
        if info_result.returncode == 0:
            info = json.loads(info_result.stdout)
            caps = info.get('automatic_captions', {}).get('en', [])
            for cap in caps:
                url = cap.get('url', '')
                if 'm3u8' in url:
                    print(f"  HLS caption stream found: {url[:80]}...")
                    # Fetch m3u8 index
                    import urllib.request
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    resp = urllib.request.urlopen(req, timeout=30)
                    m3u8 = resp.read().decode('utf-8')
                    seg_urls = [line.strip() for line in m3u8.split('\n')
                               if line.strip().startswith('http')]
                    print(f"  Found {len(seg_urls)} VTT segments")
                    all_text = []
                    for seg_url in seg_urls:
                        req2 = urllib.request.Request(seg_url, headers={'User-Agent': 'Mozilla/5.0'})
                        resp2 = urllib.request.urlopen(req2, timeout=30)
                        vtt = resp2.read().decode('utf-8')
                        clean = re.sub(r'<[^>]*>', ' ', vtt)
                        clean = re.sub(r'X-TIMESTAMP-MAP[^ ]*|Kind: \w+', '', clean)
                        clean = re.sub(r'^WEBVTT\s*$', '', clean, flags=re.MULTILINE)
                        lines = clean.split('\n')
                        skip = False
                        for line in lines:
                            if '-->' in line:
                                skip = True
                                continue
                            if skip:
                                skip = False
                                continue
                            stripped = line.strip()
                            if stripped:
                                all_text.append(stripped)
                    # Deduplicate consecutive
                    deduped = []
                    prev = None
                    for p in all_text:
                        if p != prev:
                            deduped.append(p)
                            prev = p
                    transcript = " ".join(deduped)
                    transcript = re.sub(r'\.{2,}', '.', transcript)
                    transcript = re.sub(r'\s+', ' ', transcript).strip()
                    if len(transcript) > 200:
                        print(f"  HLS caption fallback succeeded: {len(transcript)} chars")
                        return transcript, None
    except Exception as e:
        print(f"  HLS caption fallback error: {e}")

    # Clean up
    for f in vtt_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    error_msg = f"Primary: {err_primary}"
    if dl_result.returncode != 0:
        error_msg += f"; yt-dlp: {dl_result.stderr[:300]}"
    print(f"  Transcript fetch failed (all methods): {error_msg}")
    return None, error_msg


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
    uv_path = os.path.expanduser("~/.local/bin/uv")
    result = subprocess.run(
        [uv_path, "run", "--with", "yt-dlp", "python3", "-c", code,
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


# ── LLM calls ──────────────────────────────────────────────────────────

def call_llm(system_prompt, user_content, max_tokens=2000, thinking=False):
    """Call the LLM. Handles both content and reasoning-only models.

    thinking: when False (default) the Qwen reasoning tokens are suppressed via
    llama.cpp's chat_template enable_thinking flag. WHY: with thinking ON the
    model burns the ENTIRE max_tokens budget on reasoning (verified: 4000/4000
    reasoning_tokens, 0 content), so call_llm's fallback carves the note out of
    the reasoning tail — producing either a thinking-trace fragment or a note
    truncated mid-sentence. Disabling it returns the note in `content` directly
    and leaves the full budget for the output.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if not thinking:
        # llama.cpp knob; tested 2026-08-19: reasoning_tokens=0, finish_reason=stop
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        resp = request.urlopen(req, timeout=180)
        result = json.loads(resp.read())
        choice = result["choices"][0]
        msg = choice["message"]

        content = msg.get("content")
        from_reasoning = False
        if not content:
            from_reasoning = True
            reasoning = msg.get("reasoning", "")
            if reasoning:
                final_match = re.search(r"Final(?:[_ ]Answer)?[^\n]*\n\n(.*)", reasoning, re.DOTALL)
                if final_match:
                    content = final_match.group(1).strip()
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', reasoning.strip())
                    content = " ".join(sentences[-5:]) if len(sentences) > 1 else reasoning.strip()
        # Definitive truncation signal: the model hit the token budget while
        # writing the actual content. The note is cut mid-sentence — treat as
        # a failed attempt so the caller records a retry instead of sealing a
        # truncated note as completed.
        if not from_reasoning and choice.get("finish_reason") == "length":
            print(f"  LLM output truncated at max_tokens ({max_tokens}) — treating as failure")
            return None
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
    matching the vault's `research-deep` knowledge-note format.
    """
    # Extract entities for the LLM prompt
    entity_context = ""
    if entities["github_urls"]:
        entity_context += "\n\nGitHub References:\n" + "\n".join(f"- {u}" for u in entities["github_urls"])
    if entities["paper_arxiv"]:
        entity_context += "\n\nArXiv References:\n" + "\n".join(f"- {a}" for a in entities["paper_arxiv"])
    if entities["paper_urls"]:
        entity_context += "\n\nPaper URLs:\n" + "\n".join(f"- {u}" for u in entities["paper_urls"])

    prompt = f"""You are creating a structured knowledge note for a YouTube video.
Generate a complete vault knowledge note in the following format.

INPUT VIDEO:
- Title: {title}
- Channel: @{CHANNEL_HANDLE} ({CHANNEL_NAME})
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
channel: {CHANNEL_HANDLE}
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
    # thinking=False (default): Qwen3.8 burns the entire token budget on
    # reasoning otherwise (verified 4000/4000 reasoning_tokens, 0 content),
    # which used to produce truncated or thinking-trace notes.
    # 8192 budget: measured notes run 1.4–2k content tokens, leaving headroom
    # for the longest talks without hitting finish_reason=length.
    result = call_llm(
        "You are a knowledge engineer creating structured technical notes. Output ONLY the markdown note in the format shown — no preamble, no explanations, no conversational filler. Include frontmatter, executive summary, key points, technical details, tools, related resources, and open questions.",
        prompt,
        max_tokens=8192,
        thinking=False,
    )
    if result:
        # Qwen sometimes wraps the whole note in a ```markdown fence
        # (verified 2026-08-19). Strip a single wrapping fence if present.
        r = result.strip()
        if r.startswith("```") and r.endswith("```"):
            first_nl = r.find("\n")
            if first_nl != -1:
                r = r[first_nl + 1:]
                if r.endswith("```"):
                    r = r[:-3].rstrip()
            result = r
    return result if result else None


# ── Video processing ──────────────────────────────────────────────────

def slugify(text, max_len=80):
    slug = re.sub(r"[^\w\s-]", "", text).lower().replace(" ", "-")[:max_len]
    return re.sub(r"-+", "-", slug).strip("-")


def validate_note(text, video_id):
    """Validate LLM-generated note content before it is written to disk / marked seen.

    WHY: call_llm can return a thinking-trace fragment instead of the final
    markdown (e.g. "Let's craft 3 sentences:" with no body). The old flow wrote
    that verbatim and marked the entry completed, so the backfill loop never
    retried it. This gate makes such output a *failed* attempt (retry-eligible)
    instead of a sealed corrupt note.

    Returns (ok, reason). ok is checked by callers; reason goes into state's
    last_error for the failure path.
    """
    text = (text or "").strip()
    if len(text) < 400:
        return False, f"note too short ({len(text)} chars — likely thinking-trace leak or empty output)"
    if not text.startswith("---"):
        return False, "note has no YAML frontmatter"
    # A real note has at least one section heading. Old-format notes use
    # "## Summary"; current format uses "## Executive Summary".
    if "## Summary" not in text and "## Executive Summary" not in text:
        return False, "note has no Summary section"
    # Body evidence: real notes have "## Key Points"; the transcript-fallback
    # note (built below when the LLM returns nothing) has "## Transcript"
    # instead. Accept either — the two failures we're guarding against
    # (thinking-trace leak, empty output) have neither.
    if "## Key Points" not in text and "## Transcript" not in text and "## Technical Details" not in text:
        return False, "note has no body section (Key Points / Technical Details / Transcript)"
    return True, ""


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
    transcript, err = fetch_transcript(video_id)
    if not transcript:
        print(f"  ERROR: Could not fetch transcript: {err}")
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
        paper_url = paper_url[0] if isinstance(paper_url, tuple) else paper_url
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
channel: {CHANNEL_HANDLE}
published: {publish_date}
---

# {title}

## Summary
Processing failed — transcript below.

## Transcript
{transcript[:8000]}
"""

    # Validate BEFORE writing: the LLM can return a thinking-trace fragment
    # ("Let's craft 3 sentences:" + no body) instead of markdown. The old
    # flow wrote that verbatim and the caller marked it completed, so the
    # backfill loop never retried it. Now a bad output → False → caller
    # records a retry-eligible failure and nothing is written.
    ok, reason = validate_note(knowledge_note, video_id)
    if not ok:
        print(f"  ✗ NOTE VALIDATION FAILED ({reason}) — nothing written, will retry")
        return False

    # Write note
    video_slug = slugify(title)
    date_str = publish_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    note_filename = f"{date_str}-{video_slug}.md"
    note_path = os.path.join(VAULT_YT_DIR, note_filename)

    with open(note_path, "w") as f:
        f.write(knowledge_note)
    print(f"  Created vault note: {note_path}")
    _NOTE_INDEX.clear()

    # The path, not True: callers record it in state. They used to write a
    # guessed `video-<id>.md` that never existed on disk.
    return note_path


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


def videos_above_floor(state, all_videos):
    """The playlist prefix a tracked-from-a-date channel may still look at.

    `floor_video_id` is the oldest video `--since-days` registered. Without
    it (AI Engineer, registered with a full `--backfill`) the whole list is
    eligible, which is the old behaviour. With it, the walk stops *at* the
    floor. If the floor video has since been deleted, the walk stops after
    the last video that is in `seen` at all — never past everything we know
    into the channel's back catalogue.
    """
    floor_id = state.get("floor_video_id")
    if not floor_id:
        return list(all_videos)
    ids = [v["id"] for v in all_videos]
    if floor_id in ids:
        return all_videos[: ids.index(floor_id) + 1]
    last_seen = max((i for i, vid in enumerate(ids) if vid in state.get("seen", {})), default=-1)
    return all_videos[: last_seen + 1]


def get_next_video(state, all_videos):
    """Find next video to process.

    Priority:
      1. Unseen videos (truly new, newest first per channel ordering)
         — skip unplayable ones (premieres, region-locked, etc.)
      2. Failed videos eligible for retry — never let one failed video at the
         top of the feed block fresh uploads behind it."""
    seen = state["seen"]
    now = datetime.now(timezone.utc)
    for video in videos_above_floor(state, all_videos):
        vid_id = video["id"]
        entry = seen.get(vid_id)
        # Unseen, or previously deferred (transiently unplayable) → (re)check
        # playability. A premiere/livestream stays deferred until it airs.
        if entry is None or (isinstance(entry, dict) and entry.get("status") == "deferred"):
            playable, reason, _date = is_video_playable(vid_id)
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



# ── Session-path plumbing: bundles, state transitions, the report ──────
#
# The digest and the Lloyd eval run in a real session (workers/sources/
# youtube_digest.py) so a human can read them in Inner Voice. This script is
# the deterministic half: it knows the channel, owns seen.json, and fetches
# everything the session reads into one bundle directory per video.

REPORT_DIR = os.path.expanduser("~/obsidian/projects/lloyd/channel-eval")
TRANSCRIPT_WRAP = 100
FRONTMATTER_PROBE_BYTES = 2000
JSON_MARK = "@@JSON@@ "

# Statuses the digest worker may pick up. `fetched` is a bundle whose session
# never reported back (backend restarted under it); it is re-offered rather
# than lost, and `--fetch` reuses the bundle on disk.
QUEUEABLE_STATUSES = ("pending", "fetched")

_NOTE_INDEX: dict = {}


def _is_yyyymmdd(value):
    return isinstance(value, str) and len(value) == 8 and value.isdigit()


def emit_json(obj):
    """Machine-readable result on the last stdout line, behind a marker the
    worker looks for, so progress prints above it cost nothing."""
    print(JSON_MARK + json.dumps(obj, default=str))


def bundle_dir(video_id):
    return os.path.join(STATE_DIR, "bundles", video_id)


def wrap_transcript(text, width=TRANSCRIPT_WRAP):
    """Break a one-line transcript into ~width-character lines.

    fetch_transcript joins caption segments with spaces, so a 40-minute talk
    arrives as one 60 kB line. The Read tool pages by *line* (2000 per call),
    so that shape is all-or-nothing and can blow the tool-result budget.
    Wrapped, the same transcript is ~600 lines and pages like any file.
    """
    import textwrap
    return "\n".join(textwrap.wrap(text or "", width=width,
                                   break_long_words=False, break_on_hyphens=False))


def _frontmatter_field(path, key):
    """One scalar out of a note's front matter, read from its head only."""
    if not path:
        return ""
    try:
        with open(path, errors="ignore") as f:
            head = f.read(FRONTMATTER_PROBE_BYTES)
    except OSError:
        return ""
    m = re.search(rf"^{re.escape(key)}:[ \t]*(.+?)[ \t]*$", head, re.M)
    return m.group(1).strip().strip("'\"") if m else ""


def note_index():
    """video_id → note path for the active channel's vault directory.

    Built once per process per directory: `--since-days --requeue` asks for
    ~300 lookups against a 600-file directory, and 300 × 600 head reads is a
    noticeable pause for no reason.
    """
    key = VAULT_YT_DIR
    if key in _NOTE_INDEX:
        return _NOTE_INDEX[key]
    index = {}
    if os.path.isdir(VAULT_YT_DIR):
        for name in sorted(os.listdir(VAULT_YT_DIR)):
            if not name.endswith(".md"):
                continue
            path = os.path.join(VAULT_YT_DIR, name)
            vid = _frontmatter_field(path, "video_id")
            if vid:
                index.setdefault(vid, path)
    _NOTE_INDEX[key] = index
    return index


def existing_note_for(video_id, entry=None):
    """The vault note already written for this video, if any.

    Prefers the path recorded in state when it exists on disk; otherwise the
    `video_id:` index of the channel's directory. The index is what makes AI
    Engineer's pre-existing notes usable: their state rows recorded a
    `video-<id>.md` path that was never written.
    """
    entry = entry if isinstance(entry, dict) else {}
    for key in ("existing_note", "youtube_note"):
        recorded = entry.get(key)
        if recorded and os.path.isfile(recorded):
            return recorded
    return note_index().get(video_id)


def target_note_path(title, publish_date):
    date_str = publish_date if _is_yyyymmdd(publish_date) else datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(VAULT_YT_DIR, f"{date_str}-{slugify(title)}.md")


def build_bundle(video_id, title="", published="", entry=None):
    """Fetch everything a digest session reads and write it under bundle_dir.

    Returns (meta, None) or (None, error). The transcript is the only hard
    requirement; a metadata or enrichment failure degrades to an empty field.
    Enrichment (GitHub clones, arXiv notes) stays here rather than in the
    session because it is deterministic plumbing and the session has no Bash.
    """
    bdir = bundle_dir(video_id)
    os.makedirs(bdir, exist_ok=True)

    metadata = None
    try:
        metadata = fetch_video_metadata(video_id)
    except Exception as e:  # noqa: BLE001 — degrade, the transcript decides
        print(f"  metadata fetch failed: {e}")
    description = ""
    if metadata:
        title = metadata.get("title") or title
        published = (metadata.get("upload_date") or "")[:8] or published
        description = metadata.get("description", "") or ""

    print("  Fetching transcript...")
    transcript, err = fetch_transcript(video_id)
    if not transcript:
        return None, f"Could not fetch transcript: {err}"
    words = len(transcript.split())
    print(f"  Transcript: {len(transcript)} chars, {words} words")

    entities = extract_entities(transcript)
    enrichment = {"github": [], "papers": []}
    for gh_url in entities["github_urls"]:
        try:
            r = clone_and_note(gh_url)
        except Exception as e:  # noqa: BLE001
            print(f"  GitHub enrichment failed for {gh_url}: {e}")
            r = None
        if r:
            enrichment["github"].append(r)
    for arxiv_id in entities["paper_arxiv"]:
        try:
            r = fetch_arxiv_paper(arxiv_id)
        except Exception as e:  # noqa: BLE001
            print(f"  arXiv enrichment failed for {arxiv_id}: {e}")
            r = None
        if r:
            enrichment["papers"].append(r)
    for paper_url in entities["paper_urls"]:
        paper_url = paper_url[0] if isinstance(paper_url, tuple) else paper_url
        try:
            r = fetch_generic_paper(paper_url)
        except Exception as e:  # noqa: BLE001
            print(f"  paper enrichment failed for {paper_url}: {e}")
            r = None
        if r:
            enrichment["papers"].append(r)

    wrapped = wrap_transcript(transcript)
    transcript_path = os.path.join(bdir, "transcript.txt")
    with open(transcript_path, "w") as f:
        f.write(wrapped + "\n")

    existing = existing_note_for(video_id, entry)
    measurements_path, measurements = capture_measurements(bdir)
    meta = {
        "channel_key": CHANNEL_KEY,
        "channel_handle": CHANNEL_HANDLE,
        "channel_name": CHANNEL_NAME,
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "published": published,
        "description": description[:3000],
        "bundle_dir": bdir,
        "meta_path": os.path.join(bdir, "meta.json"),
        "transcript_path": transcript_path,
        "transcript_chars": len(transcript),
        "transcript_words": words,
        "transcript_lines": wrapped.count("\n") + 1,
        "entities": {k: entities.get(k, []) for k in ("github_urls", "paper_arxiv", "paper_urls", "other_urls")},
        "enrichment": enrichment,
        "existing_note": existing,
        "target_note": existing or target_note_path(title, published),
        "measurements_path": measurements_path,
        "measurements_summary": measurements_summary(measurements),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta["meta_path"], "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta, None


MEASUREMENT_URLS = {
    "vllm_metrics": "http://127.0.0.1:8096/metrics",
    "dashboard": "http://127.0.0.1:8080/api/dashboard",
}
_VLLM_WANTED = (
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running", "vllm:num_requests_waiting",
)
EVAL_BASELINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "baselines")


def _fetch_text(url, timeout=6):
    return request.urlopen(request.Request(url, headers={"User-Agent": "lloyd-youtube-monitor"}),
                           timeout=timeout).read().decode("utf-8", "replace")


def _parse_vllm_metrics(text):
    """Sum the wanted Prometheus series across label sets."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in _VLLM_WANTED:
            continue
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        out[name] = out.get(name, 0.0) + val
    q, h = out.get("vllm:prefix_cache_queries_total"), out.get("vllm:prefix_cache_hits_total")
    if q:
        out["prefix_cache_hit_rate_since_boot"] = round((h or 0.0) / q, 4)
    return out


def newest_retrieval_baseline(baselines_dir=None):
    """The most recent retrieval-eval run that actually carries metrics.

    `eval/baselines/` holds several kinds of file — nightly runs, autoimplement
    checks, rebuild before/after pairs, per-item improve runs — and not all
    of them have an `overall` block (a rebuild-after file is a corpus
    description). Newest by mtime among those that do, nightly preferred
    when it is within a day of the newest, so the number the eval quotes is
    the one the dashboard and the autoimplement gate quote.
    """
    import glob
    d = baselines_dir or EVAL_BASELINES_DIR
    candidates = []
    for path in glob.glob(os.path.join(d, "*.json")):
        try:
            with open(path) as f:
                j = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        # The eval runner writes `summary: {overall, by_category}`; the autoimplement
        # ledger re-keys the overall block to top-level `overall`. Either
        # counts; a file with neither (a rebuild-after corpus description)
        # does not.
        summ = j.get("summary")
        metrics = None
        if isinstance(summ, dict):
            metrics = summ.get("overall") if isinstance(summ.get("overall"), dict) else summ
        if not metrics and isinstance(j.get("overall"), dict):
            metrics = j["overall"]
        if not isinstance(metrics, dict) or not metrics:
            continue
        candidates.append((os.path.getmtime(path), path, j, metrics))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    newest_mtime = candidates[0][0]
    pick = next((c for c in candidates
                 if os.path.basename(c[1]).startswith("nightly-") and newest_mtime - c[0] <= 86400),
                candidates[0])
    mtime, path, j, metrics = pick
    return {"path": os.path.abspath(path), "label": j.get("label") or os.path.basename(path),
            "measured_at": j.get("measured_at") or j.get("ran_at"),
            "overall": metrics}


def capture_measurements(bdir):
    """Snapshot the live numbers a digest session may need to judge a claim.

    The session's `http_fetch` refuses loopback by design (it fetches
    arbitrary web pages), so the first session that tried to check the
    prefix-cache counter could not reach it and fell back to guessing. This
    script has no such limit: it takes the snapshot at fetch time and the
    session Reads it. Best effort per source — a failure is recorded under
    `errors`, never raised, and the file is always written.
    """
    out = {"captured_at": datetime.now(timezone.utc).isoformat(), "vllm": {}, "dashboard": {},
           "retrieval_eval": {}, "errors": []}
    try:
        out["vllm"] = _parse_vllm_metrics(_fetch_text(MEASUREMENT_URLS["vllm_metrics"]))
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"vllm metrics: {e}")
    try:
        d = json.loads(_fetch_text(MEASUREMENT_URLS["dashboard"], timeout=10))
        out["dashboard"] = {k: d[k] for k in ("vllm", "workers", "host", "usage") if k in d}
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"dashboard: {e}")
    try:
        out["retrieval_eval"] = newest_retrieval_baseline() or {}
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"retrieval eval: {e}")

    path = os.path.join(bdir, "measurements.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return path, out


def measurements_summary(m):
    """One line for the prompt: the numbers most claims turn on."""
    bits = []
    v = (m or {}).get("vllm") or {}
    if "prefix_cache_hit_rate_since_boot" in v:
        bits.append(f"prefix-cache hit rate since boot {v['prefix_cache_hit_rate_since_boot'] * 100:.1f}%")
    if "vllm:kv_cache_usage_perc" in v:
        bits.append(f"KV cache {v['vllm:kv_cache_usage_perc'] * 100:.0f}% used")
    ev = ((m or {}).get("retrieval_eval") or {}).get("overall") or {}
    if ev.get("entity_hit_rate") is not None:
        bits.append(f"retrieval eval entity_hit_rate {ev['entity_hit_rate']}, doc_hit_rate {ev.get('doc_hit_rate')}")
    if (m or {}).get("errors"):
        bits.append(f"{len(m['errors'])} source(s) unavailable")
    return "; ".join(bits) or "no live numbers captured"


def load_bundle(video_id):
    """A bundle already on disk, or None. Both files must exist."""
    bdir = bundle_dir(video_id)
    meta_path = os.path.join(bdir, "meta.json")
    if not (os.path.isfile(meta_path) and os.path.isfile(os.path.join(bdir, "transcript.txt"))):
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── State transitions ────────────────────────────────────────────────


def _entry(state, video_id):
    e = state["seen"].get(video_id)
    if not isinstance(e, dict):
        e = {}
        state["seen"][video_id] = e
    return e


def mark_fetched(state, video_id, meta):
    e = _entry(state, video_id)
    e["title"] = meta.get("title") or e.get("title", "")
    e["published"] = meta.get("published") or e.get("published", "")
    e["status"] = "fetched"
    e["bundle_dir"] = meta.get("bundle_dir")
    e["fetched_at"] = meta.get("fetched_at")
    if meta.get("existing_note"):
        e["existing_note"] = meta["existing_note"]
    return e


def mark_completed(state, video_id, note_path, eval_result=None):
    e = _entry(state, video_id)
    e["status"] = "completed"
    e["youtube_note"] = note_path
    e["completed_at"] = datetime.now(timezone.utc).isoformat()
    if isinstance(eval_result, dict) and eval_result:
        e["eval"] = eval_result
    for k in ("failure_count", "last_attempt_at", "transcript_error", "last_error"):
        e.pop(k, None)
    return e


def mark_failed(state, video_id, reason):
    """A failed attempt on the session path. Retry-eligibility is decided by
    `_is_retry_eligible`, which reads `transcript_error` to tell a permanent
    failure (no captions) from a transient one — so a transcript failure is
    recorded under that key as well."""
    e = _entry(state, video_id)
    reason = (reason or "unspecified")[:500]
    e["status"] = "failed"
    e["failure_count"] = int(e.get("failure_count", 0)) + 1
    e["last_attempt_at"] = datetime.now(timezone.utc).isoformat()
    e["last_error"] = reason
    if "transcript" in reason.lower():
        e["transcript_error"] = reason
    return e


def pending_entries(state, now=None):
    """What the digest worker may take, newest first.

    Registered/pending videos, bundles whose session never reported back,
    and failed videos whose retry is due. `published` is YYYYMMDD or "";
    unknown dates sort last.
    """
    now = now or datetime.now(timezone.utc)
    rows = []
    for vid, e in state["seen"].items():
        if not isinstance(e, dict):
            continue
        st = e.get("status")
        if st in QUEUEABLE_STATUSES or (st == "failed" and _is_retry_eligible(e, now)):
            rows.append({
                "video_id": vid,
                "title": e.get("title", ""),
                "published": e.get("published", "") or "",
                "status": st,
                "failure_count": int(e.get("failure_count", 0) or 0),
                "existing_note": existing_note_for(vid, e),
            })
    rows.sort(key=lambda r: r["published"], reverse=True)
    return rows


def register_new(state, all_videos):
    """Register every unseen video above the floor as pending.

    The session path's counterpart to `get_next_video`: the same playability
    gate (premieres deferred, dead videos skipped), but it registers all of
    them so the worker can queue more than one. Returns the new ids.

    The publish date comes from the playability probe, not from the playlist
    entry, which carries none — see `is_video_playable`. Without it a new
    upload is registered undated and `pending_entries` sorts it behind every
    dated video in the backlog.
    """
    now = datetime.now(timezone.utc).isoformat()
    added = []
    for video in videos_above_floor(state, all_videos):
        vid = video["id"]
        entry = state["seen"].get(vid)
        is_deferred = isinstance(entry, dict) and entry.get("status") == "deferred"
        if entry is not None and not is_deferred:
            continue
        playable, reason, upload_date = is_video_playable(vid)
        if not playable:
            if _is_transient_unplayable(reason):
                print(f"  Deferring not-yet-available video {vid}: {reason}")
                mark_video_deferred(state, vid, reason, video)
            else:
                print(f"  Skipping unplayable video {vid}: {reason}")
                mark_video_skipped(state, vid, reason)
            continue
        published = video.get("published", "") or ""
        if not _is_yyyymmdd(published):
            published = upload_date if _is_yyyymmdd(upload_date) else published
        state["seen"][vid] = {
            "title": video["title"],
            "published": published,
            "status": "pending",
            "registered_at": now,
        }
        added.append(vid)
        print(f"  Registered new video: {video['title']} ({vid})")
    save_state(state)
    return added


# ── The report note ──────────────────────────────────────────────────


def _md_cell(text, limit=140):
    text = " ".join(str(text or "").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_eval_report(state, now=None):
    """One note per channel listing every eval verdict, best first.

    Regenerated whole after each completion (it is a projection of
    seen.json, not a log), so it is always current and never duplicates.
    """
    from collections import Counter
    now = now or datetime.now(timezone.utc)
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{CHANNEL_KEY}.md")

    rows = [(vid, e) for vid, e in state["seen"].items()
            if isinstance(e, dict) and isinstance(e.get("eval"), dict)]

    def relevance(e):
        try:
            return int(e["eval"].get("relevance") or 0)
        except (TypeError, ValueError):
            return 0

    rows.sort(key=lambda ve: (relevance(ve[1]), ve[1].get("published", "") or ""), reverse=True)
    verdicts = Counter((e["eval"].get("verdict") or "unknown") for _, e in rows)
    filed = [(vid, e) for vid, e in rows if e["eval"].get("filed")]

    lines = [
        "---",
        "segment: projects",
        "type: notes",
        f"title: {json.dumps(CHANNEL_NAME + ' — what it offered Lloyd')}",
        "tags:",
        "- projects",
        "- lloyd",
        "- channel-eval",
        f"- {CHANNEL_KEY}",
        f"updated: '{now.strftime('%Y-%m-%dT%H:%M:%S')}'",
        "---",
        "",
        f"# {CHANNEL_NAME} — what it offered Lloyd",
        "",
        f"Every video from @{CHANNEL_HANDLE} that went through the digest session, with the "
        f"session's verdict on whether it holds something that would improve Lloyd. "
        f"Regenerated by `scripts/youtube_channel_monitor.py --channel {CHANNEL_KEY} --eval-report` "
        f"after each completion; the verdicts live in `{STATE_FILE}`.",
        "",
        f"**{len(rows)} evaluated** — " + ", ".join(f"{k}: {v}" for k, v in sorted(verdicts.items()))
        + f". **{len(filed)} backlog draft(s) filed.**",
        "",
    ]
    if filed:
        lines += ["## Filed", ""]
        for vid, e in filed:
            ev = e["eval"]
            lines.append(f"- **#{ev['filed']}** — {_md_cell(ev.get('idea'), 200)} "
                         f"(from [{_md_cell(e.get('title'), 80)}](https://www.youtube.com/watch?v={vid}))")
        lines.append("")
    lines += [
        "## All verdicts",
        "",
        "| Rel | Verdict | Video | Published | Idea | Filed |",
        "|---:|---|---|---|---|---|",
    ]
    for vid, e in rows:
        ev = e["eval"]
        pub = e.get("published", "") or ""
        pub = f"{pub[:4]}-{pub[4:6]}-{pub[6:]}" if _is_yyyymmdd(pub) else pub
        filed_cell = f"#{ev['filed']}" if ev.get("filed") else (
            f"dup of {ev['duplicate_of']}" if ev.get("duplicate_of") else "")
        lines.append(
            f"| {relevance(e)} | {ev.get('verdict') or ''} "
            f"| [{_md_cell(e.get('title'), 80)}](https://www.youtube.com/watch?v={vid}) "
            f"| {pub} | {_md_cell(ev.get('idea'))} | {filed_cell} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def process_next(state, all_videos):
    """Handle one video and return an outcome: 'processed', 'failed' or 'caught_up'.

    Priority: a failed video whose retry is due → a registered pending video
    (newest first) → the next unseen video below the floor. The stdout markers
    (`✓ Processed:`, `✗ Failed:`, `Channel caught up.`) are what the wrapper
    skill parses — keep them.
    """
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
    now = datetime.now(timezone.utc)
    retry_candidates = []
    for vid, info in state["seen"].items():
        if isinstance(info, dict) and info.get("status") == "failed" and _is_retry_eligible(info, now):
            retry_candidates.append((vid, info))

    if retry_candidates:
        vid, info = retry_candidates[0]
        title = info.get("title", "Unknown")
        print(f"Retrying failed video: {title} ({vid})")
        note_path = process_video(vid, title, info.get("published", ""))
        prior_count = info.get("failure_count", 0)
        state["seen"][vid]["status"] = "completed" if note_path else "failed"
        if not note_path:
            state["seen"][vid]["failure_count"] = prior_count + 1
            state["seen"][vid]["last_attempt_at"] = now.isoformat()
        else:
            state["seen"][vid].pop("failure_count", None)
            state["seen"][vid].pop("last_attempt_at", None)
            state["seen"][vid]["youtube_note"] = note_path
        save_state(state)
        if note_path:
            print(f"\n✓ Retry succeeded: {title}")
            return "processed"
        print(f"\n✗ Retry failed: {title}")
        return "failed"

    if pending_vids:
        vid, info = pending_vids[0]
        title = info.get("title", "Unknown")
        print(f"Processing backfill video: {title} ({vid})")
        note_path = process_video(vid, title, info.get("published", ""))
        state["seen"][vid]["status"] = "completed" if note_path else "failed"
        if note_path:
            state["seen"][vid]["youtube_note"] = note_path
        else:
            state["seen"][vid]["failure_count"] = info.get("failure_count", 0) + 1
            state["seen"][vid]["last_attempt_at"] = now.isoformat()
        save_state(state)
        if note_path:
            print(f"\n✓ Processed: {title}")
            return "processed"
        print(f"\n✗ Failed: {title}")
        return "failed"

    # Otherwise check for new videos
    next_video = get_next_video(state, all_videos)
    if not next_video:
        print("No new videos to process. Channel caught up.")
        return "caught_up"

    note_path = process_video(
        next_video["id"],
        next_video["title"],
        next_video.get("published", ""),
    )
    if note_path:
        state["seen"][next_video["id"]] = {
            "title": next_video["title"],
            "published": next_video.get("published", ""),
            "status": "completed",
            "youtube_note": note_path,
        }
        save_state(state)
        print(f"\n✓ Processed: {next_video['title']}")
        return "processed"

    prior = state["seen"].get(next_video["id"], {})
    failure_count = prior.get("failure_count", 0) + 1
    state["seen"][next_video["id"]] = {
        "title": next_video["title"],
        "published": next_video.get("published", ""),
        "status": "failed",
        "failure_count": failure_count,
        "last_attempt_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)
    if failure_count >= MAX_FAILURE_RETRIES:
        print(f"\n✗ Failed (giving up after {failure_count} attempts): {next_video['title']}")
    else:
        print(f"\n✗ Failed (attempt {failure_count}/{MAX_FAILURE_RETRIES}, will retry): {next_video['title']}")
    return "failed"


def _upload_date_of(video_id):
    """YYYYMMDD upload date via a full yt-dlp probe, or '' when unknown.

    The flat playlist listing carries no dates at all (verified 2026-09-08:
    upload_date/timestamp/release_timestamp all None), so a date window has
    to be resolved one video at a time. ~3 s each, paid once at registration.
    """
    try:
        meta = fetch_video_metadata(video_id)
    except Exception:
        meta = None
    return (meta or {}).get("upload_date", "") or ""


def select_since(dated_videos, cutoff_yyyymmdd, patience=2):
    """Pure window selection over (video, upload_date) pairs in playlist order.

    Returns (inside, floor_id). Walks newest → oldest and stops after
    `patience` consecutive videos older than the cutoff — the uploads
    playlist is chronological, but a premiere or a re-dated upload can sit
    out of order, so one older video is not proof the window has ended.
    An undated video is inside the window while nothing older has been seen
    (one extra note is the cheap mistake; a lost video is the expensive one)
    and is assumed older once something older has — the playlist is
    chronological, so it neither extends nor resets the run.
    """
    inside, floor_id, older_run = [], None, 0
    for video, upload_date in dated_videos:
        if upload_date and upload_date < cutoff_yyyymmdd:
            older_run += 1
            if older_run >= patience:
                break
            continue
        if not upload_date:
            if older_run:
                continue
        else:
            older_run = 0
        inside.append((video, upload_date))
        floor_id = video["id"]
    return inside, floor_id


def register_since(state, all_videos, days, requeue=False):
    """Register every video published in the last `days` days as pending, and
    set the floor so the new-video walk never goes below the window.

    `requeue` also puts *completed* videos inside the window back to pending,
    keeping the note they already have as `existing_note` so the session
    refreshes it in place instead of writing a second one. That is how a
    channel whose notes predate the session path gets the eval pass.
    """
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    print(f"Registering videos published since {cutoff} ({days} days) — probing dates…")

    dated = []
    older_run = 0
    for i, v in enumerate(all_videos):
        entry = state["seen"].get(v["id"])
        entry = entry if isinstance(entry, dict) else {}
        known = entry.get("published", "")
        if not _is_yyyymmdd(known):
            # An existing note's front matter is a free, exact date — the
            # AI Engineer state rows all carry published: "" while their
            # 600 notes carry `published: 20260908`.
            note = existing_note_for(v["id"], entry)
            known = _frontmatter_field(note, "published") if note else ""
        d = known if _is_yyyymmdd(known) else _upload_date_of(v["id"])
        dated.append((v, d))
        print(f"  [{i+1}] {d or '????????'} {v['title'][:70]}")
        # Mirror select_since's stop rule so we do not probe the whole channel.
        if d and d < cutoff:
            older_run += 1
            if older_run >= 2:
                break
        elif d:
            older_run = 0

    inside, floor_id = select_since(dated, cutoff)
    added = requeued = 0
    for v, d in inside:
        existing = state["seen"].get(v["id"])
        if isinstance(existing, dict) and existing.get("status") in ("completed", "skipped"):
            if requeue and existing.get("status") == "completed":
                note = existing_note_for(v["id"], existing)
                existing["status"] = "pending"
                if not _is_yyyymmdd(existing.get("published", "")):
                    existing["published"] = d
                if note:
                    existing["existing_note"] = note
                requeued += 1
            continue
        if isinstance(existing, dict) and existing.get("status") in ("pending", "failed", "deferred"):
            existing.setdefault("published", d)
            continue
        state["seen"][v["id"]] = {"title": v["title"], "published": d, "status": "pending"}
        added += 1
    if floor_id:
        state["floor_video_id"] = floor_id
    save_state(state)
    pending = [k for k, v in state["seen"].items() if isinstance(v, dict) and v.get("status") == "pending"]
    print(f"\n  {len(inside)} videos inside the window, {added} newly registered, "
          f"{requeued} re-queued, {len(pending)} pending in total; floor = {floor_id}")
    return {"inside": len(inside), "added": added, "requeued": requeued,
            "pending": len(pending), "floor_video_id": floor_id, "cutoff": cutoff}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="YouTube channel monitor (transcript → vault note)")
    parser.add_argument("--channel", choices=sorted(CHANNELS), default=DEFAULT_CHANNEL,
                        help=f"Which tracked channel (default: {DEFAULT_CHANNEL})")
    parser.add_argument("--json", action="store_true",
                        help="Print a machine-readable result on the last line (worker use)")

    listing = parser.add_argument_group("channel listing")
    listing.add_argument("--dry-run", action="store_true", help="Show new videos without processing")
    listing.add_argument("--list", action="store_true", help="List state")
    listing.add_argument("--backfill", action="store_true", help="Register all channel videos for backfill")
    listing.add_argument("--since-days", type=int, metavar="N",
                         help="Register videos published in the last N days and set the floor there")
    listing.add_argument("--requeue", action="store_true",
                         help="With --since-days: put completed videos in the window back to pending")
    listing.add_argument("--register-new", action="store_true",
                         help="Register unseen videos above the floor as pending (worker use)")

    legacy = parser.add_argument_group("script-path digest (no session)")
    legacy.add_argument("--process-one", action="store_true", help="Process next unprocessed video")
    legacy.add_argument("--process-all", action="store_true",
                        help="Operator: keep processing until caught up (see --max)")
    legacy.add_argument("--max", type=int, default=500, help="Upper bound for --process-all")

    session = parser.add_argument_group("session path (workers/sources/youtube_digest.py)")
    session.add_argument("--pending", action="store_true", help="List what the worker may take")
    session.add_argument("--fetch", metavar="VIDEO_ID", help="Fetch one video's bundle")
    session.add_argument("--refetch", action="store_true", help="With --fetch: ignore a bundle on disk")
    session.add_argument("--complete", metavar="VIDEO_ID", help="Record a finished session (needs --note)")
    session.add_argument("--note", metavar="PATH", help="With --complete: the note the session wrote")
    session.add_argument("--eval-json", metavar="JSON", help="With --complete: the eval verdict")
    session.add_argument("--fail", metavar="VIDEO_ID", help="Record a failed session attempt")
    session.add_argument("--reason", metavar="TEXT", help="With --fail: why")
    session.add_argument("--eval-report", action="store_true", help="Regenerate the channel's report note")
    args = parser.parse_args()

    configure(args.channel)
    ensure_dirs()
    state = load_state()

    # ── Offline modes: state and disk only, no channel listing ────────
    if args.pending:
        rows = pending_entries(state)
        for r in rows:
            print(f"  [{r['status']}] {r['published'] or '????????'} {r['title']} ({r['video_id']})")
        print(f"{len(rows)} pending")
        if args.json:
            emit_json({"ok": True, "channel": channel_info(), "pending": rows})
        return

    if args.fetch:
        vid = args.fetch
        entry = state["seen"].get(vid)
        entry = entry if isinstance(entry, dict) else {}
        meta = None if args.refetch else load_bundle(vid)
        if meta:
            print(f"Reusing bundle on disk: {meta['bundle_dir']}")
            meta["existing_note"] = existing_note_for(vid, entry)
            meta["target_note"] = meta["existing_note"] or target_note_path(meta["title"], meta["published"])
            # The transcript keeps; the numbers do not.
            mpath, m = capture_measurements(meta["bundle_dir"])
            meta["measurements_path"], meta["measurements_summary"] = mpath, measurements_summary(m)
            with open(meta["meta_path"], "w") as f:
                json.dump(meta, f, indent=2, default=str)
        else:
            print(f"\n=== Fetching bundle: {entry.get('title') or vid} ({vid}) ===")
            meta, err = build_bundle(vid, entry.get("title", ""), entry.get("published", ""), entry)
            if meta is None:
                mark_failed(state, vid, err)
                save_state(state)
                print(f"\n✗ Fetch failed: {err}")
                if args.json:
                    emit_json({"ok": False, "video_id": vid, "error": err,
                               "failure_count": state["seen"][vid].get("failure_count", 0)})
                return
        mark_fetched(state, vid, meta)
        save_state(state)
        print(f"\n✓ Fetched: {meta['title']} → {meta['bundle_dir']}")
        if args.json:
            emit_json({"ok": True, "meta": meta})
        return

    if args.complete:
        if not args.note:
            parser.error("--complete requires --note")
        eval_result = None
        if args.eval_json:
            try:
                eval_result = json.loads(args.eval_json)
            except json.JSONDecodeError as e:
                parser.error(f"--eval-json is not JSON: {e}")
        mark_completed(state, args.complete, args.note, eval_result)
        save_state(state)
        print(f"✓ Completed: {args.complete} → {args.note}")
        if args.json:
            emit_json({"ok": True, "video_id": args.complete, "note": args.note})
        return

    if args.fail:
        e = mark_failed(state, args.fail, args.reason or "unspecified")
        save_state(state)
        print(f"✗ Recorded failure for {args.fail} (attempt {e['failure_count']}): {e['last_error']}")
        if args.json:
            emit_json({"ok": True, "video_id": args.fail, "failure_count": e["failure_count"]})
        return

    if args.eval_report:
        path = write_eval_report(state)
        print(f"Wrote {path}")
        if args.json:
            emit_json({"ok": True, "report": path})
        return

    # ── Online modes: need the channel listing ────────────────────────
    print("Fetching channel videos...")
    all_videos = fetch_channel_videos(limit=600 if args.since_days else 200)
    print(f"Found {len(all_videos)} videos on channel")

    if args.register_new:
        added = register_new(state, all_videos)
        rows = pending_entries(state)
        print(f"  {len(added)} newly registered; {len(rows)} pending")
        if args.json:
            emit_json({"ok": True, "channel": channel_info(), "registered": added, "pending": rows})
        return

    if args.list:
        print(f"\nState has {len(state['seen'])} seen videos:")
        for vid, info in state["seen"].items():
            print(f"  {vid}: {info.get('title', '?')} — {info.get('status', '?')}")
        return

    if args.dry_run:
        seen_ids = set(state["seen"].keys())
        for v in all_videos:
            status = "NEW" if v["id"] not in seen_ids else "seen"
            print(f"  [{status}] {v.get('published', '?')} {v['title']} ({v['id']})")
        return

    # Bounded backfill: newest → oldest, stop once past the window
    if args.since_days is not None:
        result = register_since(state, all_videos, args.since_days, requeue=args.requeue)
        if args.json:
            emit_json({"ok": True, "channel": channel_info(), **result})
        return

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

    # Script-path digest: one video (default), or drain with --process-all
    limit = args.max if args.process_all else 1
    done = 0
    while done < limit:
        outcome = process_next(state, all_videos)
        if outcome == "caught_up":
            break
        done += 1
    if done and args.process_all:
        print(f"\n--process-all: {done} video(s) handled this run")


if __name__ == "__main__":
    main()
