"""Vault writer module for storing scored items to Obsidian vault."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from . import state as scanner_state
from .models import ScoredItem, GRADE_CALL_CAP, GRADE_KEYWORD
from .profile import load_profile, get_all_keywords, keyword_match
from .body import clean_body


from ._paths import VAULT_ROOT, KNOWLEDGE_DIR, FEEDS_DIR as SCORED_FEEDS_DIR, VAULT_WRITTEN_STATE


# ── the channel monitor's note tree (backlog #1269) ─────────────────────────
#
# `scripts/youtube_channel_monitor.py` writes one real note per video under
# `knowledge/youtube/{Channel}/YYYYMMDD-slug.md`: `type: video-note` (on newer
# notes; older ones carry `type: notes` above a second front-matter block), an
# 8-14 KB Executive Summary, and the video's id in front matter as `video_id:`.
# That tree is the canonical copy of a YouTube video: it is the one with the
# transcript in it. This writer's digest is an index, not a second copy — until
# 2026-09-20 it wrote a title + `(No description)` stub beside a `watch?v=` link
# for a video it had no idea was already noted. On 2026-09-19 that was 5 of the 5
# videos the run wrote; since RELEVANCE_FLOOR landed, 7 of the 8 that cleared it.
YOUTUBE_NOTE_TREE_DIRNAME = "youtube"
_HEAD_BYTES = 4000  # same window workers/sources/youtube_digest.py matches on
_WATCH_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})")
_VIDEO_ID_RE = re.compile(r"^video_id:\s*(\S+)\s*$", re.MULTILINE)
# video_id -> note path, per knowledge dir, for the life of the process.
_NOTE_INDEX: Dict[str, Dict[str, Path]] = {}


# Items scoring below this are not written anywhere.
#
# There was no floor until 2026-09-11 (backlog #570): `write_all_to_vault()`
# appended every scored item regardless of relevance, and since the scorer could
# only ever produce 1 or 10, ~85% of what reached the vault was 1/10 noise.
# `knowledge/feeds/youtube-uncategorized.md` reached 2,545 sections / 493 KB
# that way, growing one entry per item per day. `interests.md` said a
# non-matching item "gets ignored"; the floor is what makes that sentence true.
RELEVANCE_FLOOR = 4


def below_floor(item: ScoredItem) -> bool:
    """True when an item is too weak to write to the vault."""
    try:
        return int(item.relevance) < RELEVANCE_FLOOR
    except (TypeError, ValueError):
        return True  # an unscored item is not evidence of relevance


def refused_by_call_cap(item: ScoredItem) -> bool:
    """True when the stage-2 model was never consulted about this item.

    The floor cannot do this job, and for a specific arithmetic reason: no topic
    in `interests.md` sets `**Weight:**`, so `_keyword_fallback` scores any
    whole-word match the loader's 1.0 default × 10 = 10. Measured over
    `intel-2026-09-22.jsonl`, the 19 items `RELEVANCE_FLOOR` held were all
    model-graded and 0 were ungraded — an ungraded item scores exactly 10 or
    exactly 1, so there is nothing between 4 and 10 for a floor to bite on, and 101
    ungraded items went into the vault at `Relevance: 10/10`.

    So the refusal is by *cause* rather than by magnitude, and it is deliberately
    narrow: only `GRADE_CALL_CAP` — eligible for a call, but the budget was spent
    before the model got there — is barred. `GRADE_NO_USABLE_GRADE` (asked, junk
    came back) and `GRADE_KEYWORD` (engine off, e.g. `INTEL_DISABLE_LLM=1`) keep
    writing, because refusing those would turn an engine outage into a zero-write
    day, and surviving one is exactly what the keyword fallback is for.
    """
    return getattr(item, "grade_source", GRADE_KEYWORD) == GRADE_CALL_CAP


# ── how old an item may be before the vault refuses it (backlog #1379) ────────
#
# There was no age bound at all until 2026-09-23: `write_all_to_vault` filtered on
# `refused_by_call_cap` and `below_floor` only, and the one date it read was
# `datetime.utcnow()` turned into the `## <today>` heading — so an item's age was
# invisible to the writer and to the reader alike. Measured by re-fetching each feed
# the 2026-09-22 run wrote (18 feeds, 0 failures, all 102 ids resolved): 7 videos were
# ≤2 days old, 23 were 3-7 d, 23 were 8-14 d, 21 were 15-30 d, 28 were >30 d, and the
# oldest was 453 days old. Every one of them was appended to a `knowledge/` digest as
# an `URGENT` / `Relevance: 10/10` entry under that morning's date.
#
# 30 days is a declared window, not a tuned one: it admits the whole normal
# distribution of a working day (71% of that bad day was already older than a week,
# but the oldest of its first four buckets was 30 d) and refuses the tail that made
# the run's output untrustworthy. Anything older belongs to the channel monitor,
# which writes the real per-video note.
MAX_ITEM_AGE_DAYS = 30

# The volume that makes an empty dedup state a claim rather than a quiet day.
#
# `state.load_state()` returns `{"seen": set()}` with no signal when the file is
# absent or empty, and nothing downstream bounded volume, so state loss and a busy day
# were the same observable. Measured on the state-loss day: 258 scored items from 1007
# raw. A normal day scores 9 from 49. The floor sits between the two so a genuinely
# big day still writes; what it cannot do is let an emptied `seen` list publish the
# entire backlog as today's news.
STATE_LOSS_MIN_SCORED_ITEMS = 100


def _parse_published(value: Any) -> Optional[datetime]:
    """The item's own publish date as an aware datetime, or None if there is none.

    None is the important answer, not an error: a feed that omits `atom:published`, a
    GitHub row, or a row written before the field existed all mean "no date", and the
    gate below must treat that as inside the window rather than as zero age or as old.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def item_age_days(item: ScoredItem,
                  now: Optional[datetime] = None) -> Optional[int]:
    """Whole days between the item's own publish date and now; None when undated.

    Floors rather than rounds, so an item exactly `MAX_ITEM_AGE_DAYS` old is inside
    the window and one day past it is outside. A future-dated item is negative, which
    is inside: clock skew must never become a held item.
    """
    published = _parse_published(getattr(item, "published", ""))
    if published is None:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    return int((now - published).total_seconds() // 86400)


def too_old_for_vault(item: ScoredItem, now: Optional[datetime] = None) -> bool:
    """True only when the item's own publish date is past the declared window."""
    age = item_age_days(item, now)
    return age is not None and age > MAX_ITEM_AGE_DAYS


def seen_id_count() -> int:
    """How many ids the scanner's dedup state holds: 0 when absent, empty or unreadable."""
    try:
        return len(scanner_state.load_state().get("seen", set()))
    except Exception as exc:  # unreadable state must not read as a clean day
        print(f"  Dedup state unreadable ({type(exc).__name__}: {exc}) — reading it as"
              f" empty, which the guard below turns into a blocked day")
        return 0


def scored_item_count_on_disk(date_str: str) -> int:
    """How many rows `intel-<date>.jsonl` holds, without parsing them."""
    path = SCORED_FEEDS_DIR / f"intel-{date_str}.jsonl"
    try:
        with open(path, "r") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def state_loss_detected(scored_count: int, date_str: str) -> bool:
    """True when an empty dedup state plus an oversized day means the state is gone.

    The only freshness mechanism this pipeline has is the `seen` list, and on
    2026-09-22 it was empty at run start (the written list held exactly that day's
    1007 ids, so it had accumulated nothing). Every video the feed still served was
    therefore "new", including a 453-day-old one. The same shape is reachable without
    any tree loss: a canary or gate run points `LLOYD_DATA` at a fresh home while
    `worktree.py` symlinks the LIVE vault, so the write stage starts with an empty
    state and a real vault to write into.
    """
    if seen_id_count() > 0:
        return False
    if scored_count <= STATE_LOSS_MIN_SCORED_ITEMS:
        return False
    print(f"STATE LOSS: the dedup state ({scanner_state.STATE_FILE}) holds 0 seen ids "
          f"while intel-{date_str}.jsonl holds {scored_count} scored items, over the "
          f"{STATE_LOSS_MIN_SCORED_ITEMS}-item floor "
          f"(vault_writer.STATE_LOSS_MIN_SCORED_ITEMS). An emptied `seen` list makes "
          f"the whole backlog look like today's news — on 2026-09-22 that wrote 102 "
          f"videos into the digests, 28 of them more than 30 days old. Writing NOTHING "
          f"to the vault. Restore scanner-state.json from ~/.lloyd-data-snapshots "
          f"(scripts/backup/restore-data.sh) or re-seed it, then re-run --write.")
    return True


def load_scored_items(date_str: str) -> List[ScoredItem]:
    """Load scored items from JSONL file."""
    scored_path = SCORED_FEEDS_DIR / f"intel-{date_str}.jsonl"
    
    if not scored_path.exists():
        return []
    
    items = []
    with open(scored_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                item = ScoredItem.from_dict(data)
                items.append(item)
    
    return items


def load_written_state() -> Dict:
    """Load state of items already written to vault."""
    if not VAULT_WRITTEN_STATE.exists():
        return {"written": []}
    
    with open(VAULT_WRITTEN_STATE, "r") as f:
        return json.load(f)


def save_written_state(state: Dict):
    """Save state of items written to vault."""
    VAULT_WRITTEN_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(VAULT_WRITTEN_STATE, "w") as f:
        json.dump(state, f, indent=2)


def is_written(item_id: str, state: Dict) -> bool:
    """Check if an item has already been written to vault."""
    return item_id in state.get("written", [])


def mark_written(item_id: str, state: Dict):
    """Mark an item as written to vault."""
    if "written" not in state:
        state["written"] = []
    if item_id not in state["written"]:
        state["written"].append(item_id)


def determine_vault_path(item: ScoredItem, profile: dict) -> Path:
    """Determine the vault path for a scored item."""
    source = item.source.lower()
    category = item.category.lower()
    text = f"{item.title} {item.summary}".lower()
    
    # GitHub items
    if source == "github":
        # Extract repo name from URL or title
        if "github.com" in item.url:
            url_parts = item.url.split("/")
            if len(url_parts) >= 5:
                repo_name = url_parts[4].lower().replace(".git", "")
            else:
                repo_name = "unknown-repo"
        else:
            repo_name = "unknown-repo"
        
        # Determine type from source_tags
        is_pr = "pr" in [t.lower() for t in item.source_tags]
        is_release = "release" in [t.lower() for t in item.source_tags]
        
        if is_release:
            return KNOWLEDGE_DIR / "tools" / repo_name / "releases.md"
        elif is_pr:
            return KNOWLEDGE_DIR / "tools" / repo_name / "prs.md"
        else:
            return KNOWLEDGE_DIR / "tools" / repo_name / "updates.md"
    
    # YouTube items
    elif source == "youtube":
        # Match to interest profile topics
        matched_topics = keyword_match(text, profile)
        if matched_topics:
            # Use highest weighted topic
            topic_name = max(matched_topics, key=lambda x: x["weight"])["name"]
            topic_slug = topic_name.lower().replace("_", "-")
            return KNOWLEDGE_DIR / topic_slug / "youtube-digest.md"
        else:
            return KNOWLEDGE_DIR / "feeds" / "youtube-uncategorized.md"
    
    # arXiv items
    elif source == "arxiv":
        matched_topics = keyword_match(text, profile)
        if matched_topics:
            topic_name = max(matched_topics, key=lambda x: x["weight"])["name"]
            topic_slug = topic_name.lower().replace("_", "-")
            return KNOWLEDGE_DIR / topic_slug / "papers.md"
        else:
            return KNOWLEDGE_DIR / "feeds" / "arxiv-uncategorized.md"
    
    # Hacker News items
    elif source == "hackernews":
        matched_topics = keyword_match(text, profile)
        if matched_topics:
            topic_name = max(matched_topics, key=lambda x: x["weight"])["name"]
            topic_slug = topic_name.lower().replace("_", "-")
            return KNOWLEDGE_DIR / topic_slug / "news.md"
        else:
            return KNOWLEDGE_DIR / "feeds" / "hn-uncategorized.md"
    
    # Fallback
    else:
        return KNOWLEDGE_DIR / "feeds" / "uncategorized.md"


def youtube_video_id(item: ScoredItem) -> Optional[str]:
    """The item's YouTube video id: from `watch?v=` in the URL, else the item id tail.

    `item.id` is `youtube:{channel_id}:{video_id}` (scanners/youtube_scanner.py), so
    a record whose URL carries no `watch?v=` — a feed URL, a rewriter — still
    resolves instead of silently falling through to the duplicate path.
    """
    match = _WATCH_RE.search(item.url or "")
    if match:
        return match.group(1)
    item_id = item.id or ""
    if item_id.startswith("youtube:") and ":" in item_id[len("youtube:"):]:
        tail = item_id.rsplit(":", 1)[1].strip()
        return tail or None
    return None


def _scan_note_tree(notes_dir: Path) -> Dict[str, Path]:
    """Map every `video_id:` declared under the note tree to the note that declares it.

    Keyed on front matter, never the filename: an id in a slug is not evidence that
    the note is *for* that video (25 ids here are declared by two files at once, and
    103 of the 756 declaring files carry a `type:` other than `video-note`, so
    neither the name nor the type can be the key). On a collision the
    note with the most bytes wins — that is the one with the transcript in it — with
    mtime and path as deterministic tie-breaks.
    """
    candidates: Dict[str, List[tuple]] = {}
    try:
        paths = [p for p in notes_dir.rglob("*.md") if p.is_file()]
    except OSError:
        return {}
    for path in paths:
        try:
            head = path.read_bytes()[:_HEAD_BYTES].decode("utf-8", "replace")
        except OSError:
            continue
        match = _VIDEO_ID_RE.search(head)
        if not match:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        candidates.setdefault(match.group(1), []).append(
            (stat.st_size, stat.st_mtime, str(path), path))
    return {video_id: max(entries)[3] for video_id, entries in candidates.items()}


def _note_index(notes_dir: Path, refresh: bool = False) -> Dict[str, Path]:
    """The note tree's video_id index, built once per process per knowledge dir.

    A miss re-scans once before it is believed: the channel monitor writes notes from
    another process, so "not indexed yet" must not be reported as "no note" — that
    would write the duplicate this exists to prevent.
    """
    key = str(notes_dir)
    if refresh or key not in _NOTE_INDEX:
        _NOTE_INDEX[key] = _scan_note_tree(notes_dir)
    return _NOTE_INDEX[key]


def resolve_video_note(item: ScoredItem) -> Optional[Path]:
    """The canonical per-video note for a scored YouTube item, or None."""
    if (item.source or "").lower() != "youtube":
        return None
    video_id = youtube_video_id(item)
    if not video_id:
        return None
    notes_dir = KNOWLEDGE_DIR / YOUTUBE_NOTE_TREE_DIRNAME
    index = _note_index(notes_dir)
    if video_id not in index:
        index = _note_index(notes_dir, refresh=True)
    return index.get(video_id)


def _inside_note_tree(path: Path) -> bool:
    """True when a computed target lands in the canonical note tree."""
    notes_dir = KNOWLEDGE_DIR / YOUTUBE_NOTE_TREE_DIRNAME
    try:
        target = Path(os.path.realpath(path))
        root = Path(os.path.realpath(notes_dir))
    except OSError:
        return False
    return target == root or root in target.parents


def _entry_body(item: ScoredItem) -> str:
    """What goes under the Source/Relevance line: the summary, then the scorer's reason.

    `(No description)` used to be the whole body whenever `summary` was empty — and
    stage 1 leaves `summary` empty for every YouTube item (backlog #1155), while
    stage 2 fills `why`. So the placeholder was the default, not the exception: all 8
    post-floor YouTube records since 2026-09-11 carry an empty `summary` and a
    populated `why`, which the writer never read.

    A GitHub summary goes through `body.clean_body` first (backlog #1225): an
    unfilled PR template or a body under the floor is not pasted but named, as
    `None — <reason>`, and a body that only restates the title is left out. That is
    the one case this returns "", which the caller renders as no body line. Other
    sources are not upstream templates, and a short feed summary is still a summary.
    """
    summary = (item.summary or "").strip()
    if summary and (item.source or "").lower() != "github":
        return summary
    if summary:
        body, reason = clean_body(summary, item.title or "")
        if body:
            return body
        return f"None — {reason}" if reason else ""
    why = (getattr(item, "why", "") or "").strip()
    if why:
        return why
    return "(No description)"


# ── a body that is nothing but trailers (backlog #1509) ──────────────────────
#
# `clean_body` keeps what a commit message says beyond its subject line, and a
# message whose only other line is `Co-authored-by: Name <mail>` clears the
# 40-character floor on the trailer alone. On 2026-09-25 that published
# `knowledge/tools/openclaw/updates.md` → `### refactor(llm-core): …` whose whole
# body was one `Co-authored-by:` line — noise a reader cannot tell from a note.
# Such an item is not written at all, and the run summary counts it.
_TRAILER_LINE_RE = re.compile(
    r"^\s*(?:co-authored-by|signed-off-by|reviewed-by|acked-by|tested-by|"
    r"reported-by|suggested-by|helped-by|cc|change-id)\s*:.*$",
    re.IGNORECASE)


def is_trailer_only(body: str) -> bool:
    """True when every non-blank line of `body` is a git trailer, and there is one."""
    lines = [line for line in (body or "").splitlines() if line.strip()]
    return bool(lines) and all(_TRAILER_LINE_RE.match(line) for line in lines)


def lacks_body(item: ScoredItem) -> bool:
    """True when the entry this item would render carries no prose at all.

    A video that already has a note renders a pointer, which is a body; every
    other item is judged on what `_entry_body` would write.
    """
    if resolve_video_note(item) is not None:
        return False
    return is_trailer_only(_entry_body(item))


def _note_pointer(item: ScoredItem, note: Path, digest_path: Path) -> str:
    """The body for a video that already has a note: a pointer, never a stub.

    The vault-relative path is named in prose so a reader (and this item's
    verification grep) can find the canonical note, and linked relatively so Obsidian
    resolves it from the digest it is written into.
    """
    try:
        display = note.resolve().relative_to(VAULT_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        display = note.as_posix()
    try:
        link = os.path.relpath(note.resolve(), digest_path.parent.resolve())
    except (ValueError, OSError):
        link = display
    why = (getattr(item, "why", "") or "").strip()
    reason = f" — {why}" if why else ""
    return (f"**Already noted:** [{display}]({link}){reason}\n\n"
            f"The YouTube channel monitor holds the full note for this video; this "
            f"digest indexes it instead of restating it.")


def url_exists_in_file(url: str, file_path: Path) -> bool:
    """Check if a URL already exists in the file (simple dedup)."""
    if not file_path.exists():
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return url in content
    except Exception:
        return False


def _digest_target(item: ScoredItem, vault_path: Path) -> Path:
    """Never let a digest land inside the canonical note tree.

    `determine_vault_path` derives the directory from a profile topic slug, and the
    topic name is free text: a topic named `YouTube` slugifies to `youtube`, the note
    tree's own directory name, and the writer would create a `youtube-digest.md` in
    among the notes it is supposed to be indexing. Such an entry goes to the no-match
    feed file instead, which is where an unrouted item already goes.
    """
    if _inside_note_tree(vault_path):
        return KNOWLEDGE_DIR / "feeds" / f"{item.source.lower()}-uncategorized.md"
    return vault_path


def write_item_to_vault(item: ScoredItem, profile: dict) -> bool:
    """Write a single scored item to the vault."""
    # Both new guards run here as well as in the batch filter above
    # `write_all_to_vault`, because this is the other public route into the vault
    # and a guard on one of two write surfaces is not a guard: the batch filter
    # prints the count, but any caller that reaches this function directly would
    # otherwise get a stale item — or an entire backlog after state loss — written
    # with no filter in its way.
    #
    # This route takes no date, so the day it counts is the day of the call — the
    # same key the `## <today>` heading further down uses, hoisted so the two cannot
    # disagree across midnight UTC.
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if state_loss_detected(scored_item_count_on_disk(today), today):
        return False

    if refused_by_call_cap(item):
        print(f"  Refusing (never asked about): {item.url}")
        return False

    age = item_age_days(item)
    if age is not None and age > MAX_ITEM_AGE_DAYS:
        print(f"  Held ({age} d old, past the {MAX_ITEM_AGE_DAYS}-day freshness window"
              f" vault_writer.MAX_ITEM_AGE_DAYS): {item.url}")
        return False

    vault_path = _digest_target(item, determine_vault_path(item, profile))

    # Ensure parent directory exists
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for duplicates
    if url_exists_in_file(item.url, vault_path):
        print(f"  Skipping (already exists): {item.url}")
        return False

    # The other dedup layer, which the two checks above cannot see: both inspect
    # only this item's own id and this one target file, so neither knows that a
    # different writer already put this video in `knowledge/youtube/**`.
    note = resolve_video_note(item)

    # Prepare content
    today = datetime.utcnow().strftime("%Y-%m-%d")
    source = item.source
    relevance = item.relevance
    category = item.category or "general"
    
    # Build frontmatter tags
    tags = ["intel-pipeline", source, category]
    
    # Format content - just the entry, not the full file
    body = _note_pointer(item, note, vault_path) if note else _entry_body(item)
    # The batch filter counts these; this is the other public route in (#1509).
    if not note and is_trailer_only(body):
        print(f"  Skipping (no body — trailer only): {item.url}")
        return False
    body_block = f"{body}\n\n" if body else ""

    content = f"""## {today}

### {item.title}

**Source:** {source} | **Relevance:** {relevance}/10

{body_block}[Link]({item.url})

---

"""
    
    # Append or create file
    if vault_path.exists():
        # Append to existing file
        with open(vault_path, "a") as f:
            f.write(content)
    else:
        # Create new file with frontmatter and header
        header = f"""---
segment: knowledge
type: notes
tags:
{chr(10).join(f'  - {tag}' for tag in tags)}
---

# {category.title()} Updates

"""
        with open(vault_path, "w") as f:
            f.write(header + content)
    
    print(f"  Written: {vault_path.relative_to(VAULT_ROOT)}")
    return True


def write_all_to_vault(date_str: Optional[str] = None) -> int:
    """
    Write all scored items for a date to the vault.
    
    Args:
        date_str: Date string in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Number of items written
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    print(f"\n=== Vault Writer for {date_str} ===\n")
    
    # Load scored items
    items = load_scored_items(date_str)
    if not items:
        print(f"No scored items found for {date_str}")
        return 0
    
    print(f"Loaded {len(items)} scored items")

    # State loss first, before anything is written or claimed as written. An emptied
    # `seen` list is not a quiet day: it is the whole backlog arriving at once, and
    # the run that met it on 2026-09-22 published 102 videos, 28 of them over 30 days
    # old. Returning before `save_written_state` matters — a partial write here would
    # tell the next run the day had been published.
    if state_loss_detected(len(items), date_str):
        return 0

    # The call-cap refusal runs *before* the floor, and says its own number: an
    # item the model was never asked about has no relevance to measure, so letting
    # it reach a relevance comparison is what put 101 ungraded items in the vault
    # on 2026-09-22 (see refused_by_call_cap).
    gradeable = [item for item in items if not refused_by_call_cap(item)]
    cap_refused = len(items) - len(gradeable)
    if cap_refused:
        print(f"Held {cap_refused} item(s) the stage-2 model was never asked about: "
              f"the call budget was spent before them, so their relevance is an "
              f"ungraded keyword score (grade_source={GRADE_CALL_CAP})")

    # Relevance floor: noise never enters the vault, so the feed files stop
    # growing one section per junk item per day.
    keepers = [item for item in gradeable if not below_floor(item)]
    held = len(gradeable) - len(keepers)
    if held:
        print(f"Held {held} item(s) below relevance floor {RELEVANCE_FLOOR} "
              f"(set in vault_writer.RELEVANCE_FLOOR)")

    # Freshness: an item the source published outside the window is not today's news,
    # whatever the keyword score says. The count alone would not do — the number that
    # made 2026-09-22 alarming was the age behind it, so the oldest is printed too.
    # An item with no publish date is inside the window by definition (clause 3): the
    # gate must never zero the writer on a source that carries no date.
    fresh = [item for item in keepers if not too_old_for_vault(item)]
    stale = len(keepers) - len(fresh)
    if stale:
        oldest = max(age for age in (item_age_days(item) for item in keepers)
                     if age is not None)
        print(f"Held {stale} item(s) published more than {MAX_ITEM_AGE_DAYS} days ago "
              f"(vault_writer.MAX_ITEM_AGE_DAYS): oldest {oldest} d — the item's own "
              f"publish date, not the day the scanner found it")

    # A body that is only a `Co-authored-by:`-style trailer says nothing (#1509):
    # refuse it here, where the drop is counted, rather than let it land silently.
    bodied = [item for item in fresh if not lacks_body(item)]
    no_body = len(fresh) - len(bodied)
    if no_body:
        print(f"Held {no_body} item(s) whose body is only git trailers "
              f"(vault_writer.is_trailer_only)")

    # Load written state
    written_state = load_written_state()
    
    # Load interest profile
    profile = load_profile()
    
    # Write items
    written_count = 0
    for item in bodied:
        if is_written(item.id, written_state):
            print(f"  Skipping (already written): {item.id}")
            continue

        try:
            if write_item_to_vault(item, profile):
                mark_written(item.id, written_state)
                written_count += 1
        except Exception as e:
            print(f"  Error writing {item.id}: {e}")
    
    # Save written state
    save_written_state(written_state)
    
    print(f"\n=== Vault Write Complete ===")
    print(f"Wrote {written_count} items to vault")
    print(f"skipped (no body): {no_body}")

    return written_count


if __name__ == "__main__":
    # Example usage
    count = write_all_to_vault()
    print(f"\nTotal items written: {count}")
