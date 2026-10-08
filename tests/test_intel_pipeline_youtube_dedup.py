"""Backlog #1269 — the intel-pipeline must not stub a video the channel monitor noted.

Found by autonomy task #30 on 2026-09-19 and confirmed at triage the same day. Two
scheduled jobs ingest the same YouTube RSS feeds and both write knowledge notes with
no cross-dedup: `scripts/youtube_channel_monitor.py` writes one 8-14 KB `type:
video-note` per video under `knowledge/youtube/{Channel}/`, and the intel-pipeline's
stage 4 appended a per-item block to `knowledge/{topic-slug}/youtube-digest.md`
whose body was the literal string `(No description)`.

Measured on the 2026-09-19T18:46Z run: all 5 videos it wrote (relevance 8, 8, 7, 6,
6) already had a per-video note, written earlier the same morning. Since
`RELEVANCE_FLOOR` (= 4) landed 2026-09-11, 7 of the 8 YouTube items that cleared the
floor duplicate an existing note (09-16: 2 of 2; 09-19: 5 of 5) — the floor did not
stop the duplication, it concentrated it. All 8 carry a populated `why` and an empty
`summary`, so the writer threw away text it had in hand.

Each test pins one acceptance clause. Nothing here touches the real vault or
`_pipeline`: every module holds its paths as module-level names resolved from
`Path.home()` at import, so `scratch_vault` rebinds the in-process ones and the
last test redirects HOME for the `python -m intel_pipeline` subprocess the autonomy
worker actually spawns.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INTEL_DIR = REPO_ROOT / "scripts" / "intel-pipeline"
if str(INTEL_DIR) not in sys.path:
    sys.path.insert(0, str(INTEL_DIR))

from intel_pipeline import profile as profile_mod  # noqa: E402
from intel_pipeline import vault_writer as vw_mod  # noqa: E402
from intel_pipeline.models import ScoredItem  # noqa: E402

# The 5 videos the 2026-09-19T18:46Z run wrote, with their real relevance grades and
# the real (empty) summary. Their notes are `knowledge/youtube/AI_Engineer/<file>`.
DAY = "2026-09-19"
REAL_RUN = [
    ("cQQbJqvZkpo", 8, "Vertical Mobility: Inference from MVP to Trillion-Parameter "
                       "Workloads — Sitanshu Gupta, CoreWeave",
     "20260919-vertical-mobility-inference-from-mvp-to-trillion-parameter-"
     "workloads-sitanshu-gu.md",
     "Directly addresses AI/LLM inference infrastructure and scaling, which is "
     "critical for the AI-LLM interest."),
    ("sOB3HSiG8vo", 8, "Routing LLM Inference in Production: From Engine Signals to "
                       "Policy — Qianru Lao & Lu Zhang, OpenAI",
     "20260919-routing-llm-inference-in-production-from-engine-signals-to-policy-"
     "qianru-lao-lu.md",
     "Directly addresses production engineering for LLMs, which is core to the "
     "ai-llms interest."),
    ("7c9FSUVcXR0", 7, "Operating Distributed Inference Systems at Scale — Nishant "
                       "Gupta & Naman Ahuja, Meta",
     "20260919-operating-distributed-inference-systems-at-scale-nishant-gupta-"
     "naman-ahuja-meta.md",
     "Directly addresses large-scale AI infrastructure and inference systems, which "
     "is critical for deploying LLMs and robotics AI."),
    ("Hvb2LfMH58c", 6, "The Frontier AI Inference Cloud for Agents — Byung-Gon (Gon) "
                       "Chun, FriendliAI",
     "20260919-the-frontier-ai-inference-cloud-for-agents-byung-gon-gon-chun-"
     "friendliai.md",
     "Directly addresses AI infrastructure for agents, which is critical for LLM and "
     "robotics applications, though it lacks specific hardware or voice details."),
    ("75ckHC2LU_0", 6, "What's New in Inference Engineering — Philip Kiely, Baseten",
     "20260919-whats-new-in-inference-engineering-philip-kiely-baseten.md",
     "Directly addresses AI/LLM inference engineering, which is core to the ai-llms "
     "interest, though it lacks specific robotics or voice focus."),
]
CHANNEL_ID = "UCLKPca3kwwd-B59HNr-_lvA"

# A topic in the shape `load_profile` produces (already slugified) whose keyword the
# test items carry, so `determine_vault_path` routes to knowledge/ai-llms/.
PROFILE = {
    "topics": [{"name": "ai-llms", "weight": 0.9, "keywords": ["inference"],
                "projects": [], "depth": "deep"}],
    "all_keywords": ["inference"],
    "all_projects": [],
}
# A profile topic literally named "YouTube": `slugify("YouTube")` is `youtube`, the
# note tree's own directory name, so the computed digest path lands inside it.
PROFILE_NAMED_YOUTUBE = {
    "topics": [{"name": "youtube", "weight": 0.9, "keywords": ["inference"],
                "projects": [], "depth": "deep"}],
    "all_keywords": ["inference"],
    "all_projects": [],
}


@pytest.fixture
def scratch_vault(tmp_path, monkeypatch):
    """Point every package module at tmp_path instead of the live vault/_pipeline."""
    vault = tmp_path / "obsidian"
    feeds = tmp_path / "lloyd" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    (vault / "knowledge").mkdir(parents=True)

    monkeypatch.setattr(vw_mod, "SCORED_FEEDS_DIR", feeds)
    monkeypatch.setattr(vw_mod, "VAULT_WRITTEN_STATE", feeds / "vault-written.json")
    monkeypatch.setattr(vw_mod, "KNOWLEDGE_DIR", vault / "knowledge")
    monkeypatch.setattr(vw_mod, "VAULT_ROOT", vault)
    monkeypatch.setattr(profile_mod, "PROFILE_FILE", vault / "interests.md")
    return tmp_path


def _video_note(vault: Path, video_id: str, filename: str,
                title: str = "A channel-monitor note") -> Path:
    """A per-video note in the shape `youtube_channel_monitor.py` writes."""
    path = vault / "obsidian" / "knowledge" / "youtube" / "AI_Engineer" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "segment: knowledge\n"
        "type: video-note\n"
        "domain: ai\n"
        f"source: https://www.youtube.com/watch?v={video_id}\n"
        f"video_id: {video_id}\n"
        "channel: aiDotEngineer\n"
        "published: 20260919\n"
        "---\n\n"
        f"# {title}\n\n"
        "## Executive Summary\n\n"
        + ("A real transcript summary of eight to fourteen kilobytes. " * 40)
        + "\n",
        encoding="utf-8")
    return path


def _yt_item(video_id: str, relevance: int = 6, summary: str = "",
             why: str = "Matches: inference",
             title: str = "An inference engineering panel",
             url: str | None = None,
             gate_description: str = "") -> ScoredItem:
    """A scored record in the shape stage 2 writes: empty summary, populated `why`.

    `gate_description` is the pre-strip channel text #2241 keeps beside the stripped
    `summary`. "" is every non-YouTube row, and every YouTube day file written before
    #2314, whose key is simply absent.
    """
    return ScoredItem(
        id=f"youtube:{CHANNEL_ID}:{video_id}",
        source="youtube",
        title=title,
        url=url or f"https://www.youtube.com/watch?v={video_id}",
        summary=summary,
        discovered_at="2026-09-19T18:46:00+00:00",
        relevance=relevance,
        why=why,
        category="ai-infrastructure",
        gate_description=gate_description,
    )


def _digest(vault: Path, slug: str = "ai-llms") -> Path:
    return vault / "obsidian" / "knowledge" / slug / "youtube-digest.md"


def _note_tree(vault: Path) -> Path:
    return vault / "obsidian" / "knowledge" / "youtube"


def _tree_fingerprint(tree: Path) -> dict:
    """Every byte of the canonical tree, keyed by path — the note tree's identity."""
    if not tree.is_dir():
        return {}
    return {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(tree.rglob("*")) if p.is_file()}


def _blocks(path: Path) -> list:
    return re.split(r"\n## ", path.read_text(encoding="utf-8"))


# ── clause 1: a noted video gets a pointer, resolved on front matter ──────────


def test_a_video_the_monitor_already_noted_gets_a_pointer_not_a_stub(scratch_vault):
    """The duplicate entry names the note's vault path and carries no stub body."""
    note = _video_note(scratch_vault, "7c9FSUVcXR0",
                       "20260919-operating-distributed-inference-systems-at-scale-"
                       "nishant-gupta-naman-ahuja-meta.md")
    rel = note.relative_to(scratch_vault / "obsidian")

    assert vw_mod.write_item_to_vault(_yt_item("7c9FSUVcXR0", 7), PROFILE) is True

    digest = _digest(scratch_vault)
    text = digest.read_text(encoding="utf-8")
    assert str(rel) in text, f"the entry does not name the note {rel}:\n{text}"
    assert "(No description)" not in text
    assert "watch?v=7c9FSUVcXR0" in text, "the entry lost the link it indexes"
    assert len(_blocks(digest)) == 2, "one item must add exactly one entry"


def test_resolution_reads_front_matter_and_never_the_filename(scratch_vault):
    """A video id in a filename means nothing; the `video_id:` line is the key."""
    note = _video_note(scratch_vault, "REALID99xyz",
                       "20260919-DEADBEEF404-notes-some-other-video.md")
    note_rel = str(note.relative_to(scratch_vault / "obsidian"))

    vw_mod.write_item_to_vault(_yt_item("DEADBEEF404", 6), PROFILE)
    vw_mod.write_item_to_vault(_yt_item("REALID99xyz", 6), PROFILE)

    digest = _digest(scratch_vault)
    first, second = _blocks(digest)[1:]
    assert note_rel not in first, "an id that only appears in a filename deduped"
    assert "(No description)" not in first, "a first-time write lost its `why` body"
    assert note_rel in second, "the declared video_id did not resolve to its note"


def test_the_video_id_falls_back_to_the_item_id_when_the_url_carries_none(scratch_vault):
    """`item.id` is `youtube:{channel}:{video_id}`; a URL without `watch?v=` still dedups."""
    note = _video_note(scratch_vault, "cQQbJqvZkpo", "20260919-vertical-mobility.md")
    item = _yt_item("cQQbJqvZkpo", 8,
                    url="https://www.youtube.com/feeds/videos.xml?channel_id="
                        + CHANNEL_ID)

    vw_mod.write_item_to_vault(item, PROFILE)

    text = _digest(scratch_vault).read_text(encoding="utf-8")
    assert str(note.relative_to(scratch_vault / "obsidian")) in text
    assert "(No description)" not in text


# ── clause 2: a first-time write keeps today's block exactly ─────────────────


def test_a_video_with_no_note_keeps_the_current_digest_block(scratch_vault):
    """Same target path, same `## date` / `### title` / Source / Relevance / Link."""
    # The title carries the profile keyword so routing to knowledge/ai-llms/ is
    # determined by the fixture and not by which file the entry lands in.
    item = _yt_item("newvideo1", 6, summary="A real transcript summary.",
                    title="Inference for a video nobody has noted yet")
    today = datetime.utcnow().strftime("%Y-%m-%d")

    assert vw_mod.write_item_to_vault(item, PROFILE) is True

    digest = _digest(scratch_vault)
    expected = (
        f"## {today}\n\n"
        f"### {item.title}\n\n"
        "**Source:** youtube | **Relevance:** 6/10\n\n"
        "A real transcript summary.\n\n"
        f"[Link]({item.url})\n\n"
        "---\n\n"
    )
    assert expected in digest.read_text(encoding="utf-8")


def test_an_empty_summary_falls_back_to_the_scorers_own_reason(scratch_vault):
    """All 8 post-floor YouTube records carry a populated `why` and an empty summary."""
    item = _yt_item("newvideo2", 7, summary="",
                    why="Directly addresses large-scale AI infrastructure.")

    vw_mod.write_item_to_vault(item, PROFILE)

    text = _digest(scratch_vault).read_text(encoding="utf-8")
    assert "Directly addresses large-scale AI infrastructure." in text
    assert "(No description)" not in text


def test_the_placeholder_survives_only_when_summary_and_why_are_both_empty(
        scratch_vault):
    """`(No description)` is the last resort, not the default body."""
    vw_mod.write_item_to_vault(_yt_item("newvideo3", 5, summary="", why=""), PROFILE)

    text = _digest(scratch_vault).read_text(encoding="utf-8")
    assert "(No description)" in text


# ── clause 3: the note tree is canonical and never written into ──────────────


def test_handling_a_duplicate_leaves_the_canonical_note_byte_identical(
        scratch_vault):
    """Dedup reads the note tree; it must never create or modify a byte of it."""
    _video_note(scratch_vault, "7c9FSUVcXR0", "20260919-operating.md")
    tree = _note_tree(scratch_vault)
    before = _tree_fingerprint(tree)
    assert len(before) == 1

    vw_mod.write_item_to_vault(_yt_item("7c9FSUVcXR0", 7), PROFILE)

    assert _tree_fingerprint(tree) == before


def test_a_topic_slug_inside_the_note_tree_writes_to_the_feed_digest(
        scratch_vault):
    """A profile topic named `YouTube` slugifies to `youtube`: that digest must not
    be created inside the note tree, so the entry goes to the no-match feed file."""
    item = _yt_item("newvideo4", 6, summary="A real transcript summary.")
    tree = _note_tree(scratch_vault)
    before = _tree_fingerprint(tree)

    assert vw_mod.write_item_to_vault(item, PROFILE_NAMED_YOUTUBE) is True

    assert not (_note_tree(scratch_vault) / "youtube-digest.md").exists()
    feed = (scratch_vault / "obsidian" / "knowledge" / "feeds"
            / "youtube-uncategorized.md")
    assert feed.exists(), "the entry went nowhere"
    assert "watch?v=newvideo4" in feed.read_text(encoding="utf-8")
    assert _tree_fingerprint(tree) == before


# ── seam: the `python -m intel_pipeline` subprocess the autonomy worker spawns ─


def _cli_home(tmp_path: Path) -> tuple:
    """A scratch HOME with the paths `_paths` and the profile resolve to."""
    home = tmp_path / "home"
    feeds = home / "lloyd-data" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    (home / "obsidian" / "knowledge").mkdir(parents=True)
    (home / "obsidian" / "interests.md").write_text(
        "---\ntitle: Interests\n---\n\n## AI & LLMs\ninference, vllm\n",
        encoding="utf-8")
    return home, feeds


def test_cli_write_links_the_five_videos_the_2026_09_19_run_stubbled(tmp_path):
    """The acceptance check, replayed across the subprocess seam: 5 scored records
    whose ids all have a note come out as 5 links, and every note is unchanged."""
    home, feeds = _cli_home(tmp_path)
    notes = []
    records = []
    for video_id, relevance, title, filename, why in REAL_RUN:
        notes.append(_video_note(home, video_id, filename, title=title))
        records.append({
            "id": f"youtube:{CHANNEL_ID}:{video_id}",
            "source": "youtube",
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "summary": "",
            "discovered_at": "2026-09-19T18:46:00+00:00",
            "relevance": relevance,
            "urgency": "morning",
            "why": why,
            "projects": [],
            "category": "ai-infrastructure",
        })
    (feeds / f"intel-{DAY}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    before = _tree_fingerprint(home / "obsidian" / "knowledge" / "youtube")

    proc = subprocess.run(
        [sys.executable, "-m", "intel_pipeline", "--write", "--date", DAY],
        cwd=str(INTEL_DIR),
        env=dict(os.environ, HOME=str(home), LLOYD_DATA=str(home / "lloyd-data")),
        capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, proc.stderr[-2000:]
    digest = home / "obsidian" / "knowledge" / "ai-llms" / "youtube-digest.md"
    text = digest.read_text(encoding="utf-8")
    for note in notes:
        path = str(note.relative_to(home / "obsidian"))
        assert path in text, f"no link to {path}\n{proc.stdout[-2000:]}"
    assert "(No description)" not in text
    for video_id, _r, _t, _f, _w in REAL_RUN:
        assert text.count(f"watch?v={video_id}") == 1
    assert _tree_fingerprint(home / "obsidian" / "knowledge" / "youtube") == before
    assert "Wrote 5 items" in proc.stdout, proc.stdout[-2000:]


# ── #2380 — the YouTube branch routes on the text stage 1 gated on ───────────
#
# Backlog #2380, measured on the 2026-10-07T23:00Z run of autonomy task #30. Since
# #2241/#2314 stage 1 gates on `item.stage1_text()` (title + `gate_description`) while
# the published body stays on the stripped `summary` — so a model-graded keep whose
# interest keyword survives only in the channel description clears `RELEVANCE_FLOOR`
# and then reaches `determine_vault_path`, which until this round scored the STRIPPED
# copy. For such an item that score is empty by construction, so it took the no-match
# branch into `knowledge/feeds/youtube-uncategorized.md`, the file whose own semantics
# read "no `interests.md` topic matched".
#
# The fixture is that run's misfiled item, `youtube:UCLKPca3kwwd-B59HNr-_lvA:
# exiwa9QbQXI`, relevance 8, `grade_source: model`, written to that feed file at
# `knowledge/feeds/youtube-uncategorized.md:31190`. Its title and its 497-character
# stripped summary carry no whole-word keyword of the real `ai-llms` topic — the
# summary says `agents`, and `match_keywords` matches a single-word keyword whole-word
# only — while the 2,543-character `gate_description` says `agent readiness`, the
# whole-word hit at offset 846 that made `keyword_match` return `[('ai-llms', 1.0)]`
# against the live profile. Both texts are verbatim from `intel-2026-10-07.jsonl`, the
# gate text trimmed to the paragraphs that carry that hit.

GATE_ONLY_TITLE = ("The Software Factory: From Bug Report to Production Code "
                   "— Davis Palmie, Factory")

#: The stored `summary` verbatim: the channel's prose, no whole-word `agent` in it.
GATE_ONLY_SUMMARY = (
    "Engineers are going from typing code to governing agents. The next step is a "
    "system that turns bug reports into production code.\n\n"
    "Davis Palmie, member of technical staff at Factory, explains the software "
    "factory: a system of agents that takes input signals like bug reports and user "
    "feedback and turns them into production code.")

#: The stored `gate_description`, trimmed to the paragraphs holding its one whole-word
#: `agent` hit plus the link footer the strip takes off the published body.
GATE_ONLY_RAW = GATE_ONLY_SUMMARY + (
    " He covers what a factory must handle, narrow like incident triage, and "
    "Factory's eight pillars of agent readiness. He closes with why token "
    "leaderboards are Goodhart's law in action.\n\n"
    "In this talk:\n"
    "• What a software factory is, and why coding was never the bottleneck\n\n"
    "My Links 🔗\n"
    "👉 More talks: https://ai.engineer\n")

#: The single keyword the real `ai-llms` topic matched on there, at its real weight.
PROFILE_AGENT = {
    "topics": [{"name": "ai-llms", "weight": 1.0, "keywords": ["agent"],
                "projects": [], "depth": "deep"}],
    "all_keywords": ["agent"],
    "all_projects": [],
}


def _uncategorized(vault: Path) -> Path:
    return (vault / "obsidian" / "knowledge" / "feeds"
            / "youtube-uncategorized.md")


def _gate_only_item(**over) -> ScoredItem:
    """The 2026-10-07 misfiled row: keyword lives only in the gate text."""
    return _yt_item("exiwa9QbQXI", 8, title=GATE_ONLY_TITLE,
                    summary=GATE_ONLY_SUMMARY, gate_description=GATE_ONLY_RAW,
                    why="Scores 8/10: a production agent system for turning bug "
                        "reports into code.", **over)


def test_the_fixture_is_the_split_this_round_is_about():
    """The fixture must keep the shape that broke: empty on the stripped copy, a
    topic match on the gate copy. Were this to stop holding, the routing test below
    would pass for the wrong reason — an item that matches both texts routes to its
    topic under the old code too."""
    assert GATE_ONLY_RAW.startswith(GATE_ONLY_SUMMARY)
    assert vw_mod.keyword_match(f"{GATE_ONLY_TITLE} {GATE_ONLY_SUMMARY}".lower(),
                                PROFILE_AGENT) == []
    matched = vw_mod.keyword_match(f"{GATE_ONLY_TITLE} {GATE_ONLY_RAW}".lower(),
                                   PROFILE_AGENT)
    assert [(t["name"], t["weight"]) for t in matched] == [("ai-llms", 1.0)]
    # `inference`, this file's own PROFILE keyword, is in neither text: the two
    # profiles cannot be satisfying the same assertion by accident.
    assert vw_mod.keyword_match(
        f"{GATE_ONLY_TITLE} {GATE_ONLY_RAW}".lower(), PROFILE) == []


def test_a_keep_whose_only_keyword_is_the_channel_description_gets_the_topic(
        scratch_vault):
    """Clause 1: `determine_vault_path` routes on the text stage 1 gated on, so the
    2026-10-07 item lands in `knowledge/ai-llms/youtube-digest.md` and never in the
    no-match feed file."""
    item = _gate_only_item()

    assert vw_mod.determine_vault_path(item, PROFILE_AGENT).parts[-2:] == (
        "ai-llms", "youtube-digest.md")

    assert vw_mod.write_item_to_vault(item, PROFILE_AGENT) is True

    digest = _digest(scratch_vault)
    assert digest.exists(), "the gate-text keyword did not route the item to its topic"
    text = digest.read_text(encoding="utf-8")
    assert "watch?v=exiwa9QbQXI" in text
    assert not _uncategorized(scratch_vault).exists(), (
        "the item also reached the file that means 'no topic matched'")


def _day_rows(scratch_vault, rows: list) -> list:
    """Put `rows` in `intel-{DAY}.jsonl` and read them back through the writer's own
    loader — the day-file JSONL boundary scan → score → write are joined by."""
    feeds = scratch_vault / "lloyd" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / f"intel-{DAY}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return vw_mod.load_scored_items(DAY)


def test_a_day_file_with_no_gate_description_key_still_routes_on_title_plus_summary(
        scratch_vault):
    """Clause 2, across the day-file boundary: a row written before #2314 has no
    `gate_description` key at all, and `ScoredItem.from_dict` must read that absence as
    "" so `stage1_text()` is `title + " " + summary`. The keyword sits in that row's
    SUMMARY and nowhere in its title, so the write reaching the topic digest is the
    summary leg of the routing text and not the title carrying it by accident."""
    day_row = {
        "id": f"youtube:{CHANNEL_ID}:pre2314one", "source": "youtube",
        "title": "Edge serving, a panel",
        "url": "https://www.youtube.com/watch?v=pre2314one",
        "summary": "An inference engineering talk.",
        "discovered_at": "2026-09-19T18:46:00+00:00",
        "relevance": 6, "urgency": "morning", "why": "Matches: inference",
        "projects": [], "category": "ai-infrastructure",
    }
    assert "gate_description" not in day_row
    assert "inference" not in day_row["title"].lower()

    items = _day_rows(scratch_vault, [day_row])

    assert len(items) == 1
    assert items[0].gate_description == ""
    assert items[0].stage1_text() == ("Edge serving, a panel "
                                      "An inference engineering talk.")
    assert vw_mod.write_item_to_vault(items[0], PROFILE) is True
    assert "watch?v=pre2314one" in _digest(scratch_vault).read_text(encoding="utf-8")
    assert not _uncategorized(scratch_vault).exists()


def test_gate_text_survives_the_day_file_and_routes_the_write(scratch_vault):
    """The seam the writer lives on: stage 2 rebuilds the row field by field, copying
    `gate_description` at `scoring.py:621`, so it has to cross `intel-<date>.jsonl` and
    `ScoredItem.from_dict` for the widened route to exist at all. One row whose keyword
    is ONLY in that field, written back through the real loader, must still land on its
    topic — a rebuild that dropped the key would send it to the fallback instead."""
    row = {
        "id": f"youtube:{CHANNEL_ID}:gaterow1", "source": "youtube",
        "title": "Edge serving, a panel",
        "url": "https://www.youtube.com/watch?v=gaterow1",
        "summary": "Talk about serving models at the edge.",
        "discovered_at": "2026-09-19T18:46:00+00:00",
        "relevance": 8, "urgency": "morning",
        "why": "Scores 8/10: a serving stack worth reading.",
        "projects": [], "category": "ai-infrastructure",
        "grade_source": "model",
        "gate_description": "How the team ships inference behind one gateway.",
    }
    assert "inference" not in (row["title"] + row["summary"]).lower()

    items = _day_rows(scratch_vault, [row])

    assert len(items) == 1
    assert items[0].gate_description == row["gate_description"]
    assert vw_mod.keyword_match(
        f"{items[0].title} {items[0].summary}".lower(), PROFILE) == []
    assert vw_mod.write_item_to_vault(items[0], PROFILE) is True
    assert "watch?v=gaterow1" in _digest(scratch_vault).read_text(encoding="utf-8")
    assert not _uncategorized(scratch_vault).exists()


def test_a_video_whose_gate_text_matches_no_topic_still_lands_uncategorized(
        scratch_vault):
    """Clause 3: the fallback keeps its meaning. Neither copy carries the keyword, so
    the no-match feed file is still the right destination, gate text included."""
    item = _yt_item("nomatch1", 6, title="Sourdough, three ways",
                    summary="A baking video.",
                    gate_description="Flour, water, salt, time. "
                                     "No talk, no demo, no tool.")

    assert vw_mod.determine_vault_path(item, PROFILE_AGENT).parts[-2:] == (
        "feeds", "youtube-uncategorized.md")

    assert vw_mod.write_item_to_vault(item, PROFILE_AGENT) is True

    feed = _uncategorized(scratch_vault)
    assert "watch?v=nomatch1" in feed.read_text(encoding="utf-8")
    assert not _digest(scratch_vault, "ai-llms").exists()


def test_the_cli_routes_a_gate_text_only_keep_under_its_topic(tmp_path):
    """The process autonomy task #30 actually spawns — `python -m intel_pipeline
    --write` with HOME and LLOYD_DATA redirected, its profile from that home's
    `interests.md` — driven with the row this round is about: keyword only in
    `gate_description`, so the old stripped-copy routing sent it to
    `knowledge/feeds/youtube-uncategorized.md`. It must land in
    `knowledge/ai-llms/youtube-digest.md`, and the fallback file must not be created at
    all. No note exists in this home, so nothing dedups into a pointer."""
    home, feeds = _cli_home(tmp_path)
    # `agent` alone, so the profile is the one keyword the fixture's gate text carries
    # whole-word and its published copy does not — the real `ai-llms` split exactly.
    (home / "obsidian" / "interests.md").write_text(
        "---\ntitle: Interests\n---\n\n## AI & LLMs\nagent\n", encoding="utf-8")
    assert vw_mod.keyword_match(f"{GATE_ONLY_TITLE} {GATE_ONLY_SUMMARY}".lower(),
                                PROFILE_AGENT) == []
    row = {
        "id": f"youtube:{CHANNEL_ID}:exiwa9QbQXI", "source": "youtube",
        "title": GATE_ONLY_TITLE,
        "url": "https://www.youtube.com/watch?v=exiwa9QbQXI",
        "summary": GATE_ONLY_SUMMARY,
        "discovered_at": "2026-10-07T23:00:00+00:00",
        "relevance": 8, "urgency": "morning",
        "why": "Scores 8/10: agents that turn bug reports into code.",
        "projects": [], "category": "ai-agents", "grade_source": "model",
        "gate_description": GATE_ONLY_RAW,
    }
    (feeds / f"intel-{DAY}.jsonl").write_text(json.dumps(row) + "\n",
                                              encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "intel_pipeline", "--write", "--date", DAY],
        cwd=str(INTEL_DIR),
        env=dict(os.environ, HOME=str(home), LLOYD_DATA=str(home / "lloyd-data")),
        capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, proc.stderr[-2000:]
    digest = home / "obsidian" / "knowledge" / "ai-llms" / "youtube-digest.md"
    assert digest.exists(), (
        f"the gate-text keyword did not route the write: {proc.stdout[-2000:]}")
    text = digest.read_text(encoding="utf-8")
    assert "watch?v=exiwa9QbQXI" in text
    assert not (home / "obsidian" / "knowledge" / "feeds"
                / "youtube-uncategorized.md").exists(), \
        "the item also reached the file that means 'no topic matched'"
    # Published copy is unchanged by the routing: the body is the stored summary, and
    # the sentence that exists only in the gate text stays out of the note.
    assert GATE_ONLY_SUMMARY in text
    assert "eight pillars of agent readiness" not in text
    assert "My Links" not in text
    assert "Wrote 1 items" in proc.stdout, proc.stdout[-2000:]
