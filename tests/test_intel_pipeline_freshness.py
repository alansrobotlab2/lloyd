"""Backlog #1379 — a feed item's own publish date must reach the vault writer, and an
emptied dedup state must stop the writer instead of producing a big day.

Found by autonomy task #30 on 2026-09-22 and confirmed at triage on 2026-09-23. Two
halves, one measured cost.

Half 1: the YouTube scanner parses ``atom:published`` (``youtube_scanner.py:232``,
stored at ``:249``, bound to a local at ``:390``) and then drops it, because
``FeedItem`` and ``ScoredItem`` had no such field and ``stage2_score`` rebuilds each
item field-by-field (``scoring.py:282-297``) — a second drop site. ``write_all_to_vault``
filtered only on ``refused_by_call_cap`` and ``below_floor``, so no age bound existed
anywhere: on 2026-09-22 an emptied ``seen`` list pushed 102 videos into the knowledge
digests, of which 28 were more than 30 days old and the oldest was 453 days old, each
appended under a ``## 2026-09-22`` heading.

Half 2: ``state.load_state()`` returns ``{"seen": set()}`` with no signal when the
state file is absent, and nothing downstream bounded volume — so state loss is
indistinguishable from a busy day (258 scored / 1007 raw on the state-loss day against
9 scored / 49 raw normally).

Each test pins one acceptance clause. Nothing here touches the real vault or
``_pipeline``: every module holds its paths as module-level names resolved at import,
so ``redirect_paths`` rebinds the in-process ones and the last test redirects ``HOME``
and ``LLOYD_DATA`` for the ``python -m intel_pipeline`` subprocess that autonomy task
#30 actually spawns.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INTEL_DIR = REPO_ROOT / "scripts" / "intel-pipeline"
if str(INTEL_DIR) not in sys.path:
    sys.path.insert(0, str(INTEL_DIR))

from intel_pipeline import profile as profile_mod  # noqa: E402
from intel_pipeline import scoring as scoring_mod  # noqa: E402
from intel_pipeline import state as state_mod  # noqa: E402
from intel_pipeline import vault_writer as vw_mod  # noqa: E402
from intel_pipeline.models import FeedItem, ScoredItem  # noqa: E402
from intel_pipeline.scanners import youtube_scanner as yt_mod  # noqa: E402

# One fixed instant for every age in this file, so "45 days old" means the same
# number to the test and to the writer's own comparison.
NOW = datetime.now(timezone.utc)

# The profile shape the loader reads. `vllm` sits under a weight-0.9 topic, so a
# title naming it is both a stage-1 survivor and a 9/10 keyword fallback.
PROFILE_MD = """---
title: Interests
---
# Interests

## Robotics
**Weight:** 0.7
**Projects:** Alfie
humanoid, actuator, gripper

## AI & LLMs
**Weight:** 0.9
**Projects:** Lloyd
**Keywords:** agent, llm, mcp, vllm
"""


def _published_iso(days_ago: int) -> str:
    """An `atom:published` value `days_ago` old, in the format YouTube serves."""
    return (NOW - timedelta(days=days_ago)).isoformat()


def _item(item_id: str, *, source: str = "youtube", title: str = "vllm agents",
          summary: str = "", relevance: int = 6, published: str = "",
          url: str = "") -> ScoredItem:
    """A scored row in the shape the scoring stage writes."""
    return ScoredItem(
        id=item_id, source=source, title=title,
        url=url or f"https://example.test/{item_id}", summary=summary,
        discovered_at=NOW.isoformat(), authors=[], source_tags=["tag"],
        relevance=relevance, urgency="morning", why="matches vllm",
        projects=[], category="ai-llms", published=published,
    )


@pytest.fixture
def redirect_paths(tmp_path, monkeypatch):
    """Point every package module at tmp_path instead of the live vault/_pipeline.

    `_paths` resolves at import time and each module imported the result by name, so
    rebinding the per-module names is what actually moves them.
    """
    vault = tmp_path / "obsidian"
    feeds = tmp_path / "lloyd-data" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    (vault / "knowledge").mkdir(parents=True)
    (vault / "interests.md").write_text(PROFILE_MD)

    monkeypatch.setattr(state_mod, "RAW_DIR", feeds / "raw")
    monkeypatch.setattr(state_mod, "STATE_FILE", feeds / "scanner-state.json")
    monkeypatch.setattr(vw_mod, "SCORED_FEEDS_DIR", feeds)
    monkeypatch.setattr(vw_mod, "VAULT_WRITTEN_STATE", feeds / "vault-written.json")
    monkeypatch.setattr(vw_mod, "KNOWLEDGE_DIR", vault / "knowledge")
    monkeypatch.setattr(vw_mod, "VAULT_ROOT", vault)
    monkeypatch.setattr(profile_mod, "PROFILE_FILE", vault / "interests.md")
    monkeypatch.setenv("INTEL_DISABLE_LLM", "1")
    return tmp_path


def feeds_dir(tmp_path) -> Path:
    return tmp_path / "lloyd-data" / "_pipeline" / "vault-derived" / "memory" / "feeds"


def write_intel_day(tmp_path, day: str, items) -> Path:
    """Write `intel-<day>.jsonl` exactly as the scoring stage does."""
    path = feeds_dir(tmp_path) / f"intel-{day}.jsonl"
    with open(path, "w") as f:
        for item in items:
            f.write(item.to_json() + "\n")
    return path


def populate_state(*ids: str) -> None:
    """Write a dedup state holding these ids, the shape `state.save_state` writes."""
    state_mod.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state_mod.STATE_FILE.write_text(json.dumps({"seen": list(ids), "last_run": None}))


DAY = "2026-09-23"


def today_str() -> str:
    """The day key `write_item_to_vault` derives for itself (it takes no date)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── clause 1 — the publish date survives model, scanner and scorer ────────────

def test_feed_item_round_trips_published_through_to_dict_and_from_dict():
    """`atom:published` has a field to land in, and a feed without one round-trips empty.

    The scanner already had the value in a local (`youtube_scanner.py:390`); what was
    missing was the field. A row whose feed omitted `published` must come back empty
    rather than raising, because clause 3 depends on a dateless item existing.
    """
    published = _published_iso(3)

    item = FeedItem(id="youtube:UC1:abc", source="youtube", title="vllm agents",
                    url="https://youtu.be/abc", summary="", discovered_at=NOW.isoformat(),
                    published=published)

    assert item.to_dict()["published"] == published
    assert FeedItem.from_dict(item.to_dict()).published == published
    # A feed that omitted the field: absence reads as empty, not as a KeyError and
    # not as "today".
    legacy = {k: v for k, v in item.to_dict().items() if k != "published"}
    assert FeedItem.from_dict(legacy).published == ""


def test_scored_item_round_trips_published_through_the_file_the_writer_reads(
        redirect_paths):
    """The scorer and the writer are separate processes, joined by `intel-<date>.jsonl`.

    `load_scored_items` is what the write stage calls, so a `published` that lives
    only in the scoring process is absent exactly where the age gate has to act. One
    row carries a publish date, one carries none — the reader must tell them apart.
    """
    published = _published_iso(5)
    write_intel_day(redirect_paths, DAY, [
        _item("yt_fresh", published=published),
        _item("gh_dateless", source="github", title="vllm release", published=""),
    ])

    reloaded = vw_mod.load_scored_items(DAY)

    assert [i.id for i in reloaded] == ["yt_fresh", "gh_dateless"]
    assert reloaded[0].published == published
    assert reloaded[1].published == ""


def test_youtube_scanner_puts_the_feeds_publish_date_on_the_raw_row(redirect_paths,
                                                                    monkeypatch,
                                                                    capsys):
    """The scanner's own drop site: an Atom `<published>` must reach `raw/<date>.jsonl`.

    Crosses the feed-parse boundary with a real Atom body rather than a stubbed dict:
    `_http_get` is the wire, so `_parse_feed_entries` and the `FeedItem` build both
    run. Two entries — one with `atom:published`, one without — because clause 3
    needs a dateless row to exist and stay dateless.
    """
    channel_id = "UCnate"
    published = "2026-09-20T10:15:30+00:00"
    atom = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:media="http://search.yahoo.com/mrss/"
      xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>yt:video:AAAA1111</id>
  <yt:videoId>AAAA1111</yt:videoId>
  <yt:channelId>{channel_id}</yt:channelId>
  <title>vllm continuous batching explained</title>
  <published>{published}</published>
  <link rel="alternate" href="https://www.youtube.com/watch?v=AAAA1111"/>
 </entry>
 <entry>
  <id>yt:video:BBBB2222</id>
  <yt:videoId>BBBB2222</yt:videoId>
  <yt:channelId>{channel_id}</yt:channelId>
  <title>vllm without a publish date</title>
  <link rel="alternate" href="https://www.youtube.com/watch?v=BBBB2222"/>
 </entry>
</feed>"""

    monkeypatch.setattr(yt_mod, "load_youtube_channels_config", lambda: [
        {"handle": "@natebjones", "name": "Nate B Jones", "channel_id": channel_id}])
    monkeypatch.setattr(yt_mod, "_http_get",
                        lambda url, headers=None, timeout=None: atom)

    items, coverage = yt_mod.scan_youtube_channels()

    assert coverage.fetched == 1
    assert [i.id for i in items] == [f"youtube:{channel_id}:AAAA1111",
                                     f"youtube:{channel_id}:BBBB2222"]
    assert items[0].published == published
    assert items[1].published == ""
    # The file, not just the object: this is the row `--score` reads back tomorrow.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw_path = state_mod.RAW_DIR / f"{today}.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    assert [r.get("published") for r in rows] == [published, ""]
    assert "Saved 2 items to raw JSONL" in capsys.readouterr().out


def test_stage2_score_copies_published_onto_every_scored_row(redirect_paths):
    """The second drop site: `stage2_score` rebuilds each item field-by-field.

    A field the scanner passes but the scorer omits still disappears from
    `intel-<date>.jsonl`, which is the only file the writer reads — so this is a
    separate failure from the scanner's, on the same clause.
    """
    published = _published_iso(2)
    raw = [
        FeedItem(id="youtube:UC1:aaa", source="youtube",
                 title="vllm speculative decoding for agents", summary="",
                 url="https://youtu.be/aaa", discovered_at=NOW.isoformat(),
                 authors=["Nate B Jones"], source_tags=["@natebjones"],
                 published=published),
        FeedItem(id="github:owner/repo:1", source="github", title="vllm 0.29.1",
                 summary="vllm release", url="https://github.com/owner/repo",
                 discovered_at=NOW.isoformat(), authors=["owner"],
                 source_tags=["release"], published=""),
    ]

    scored = scoring_mod.run_scoring_pipeline(raw, profile_mod.load_profile())

    by_id = {item.id: item for item in scored}
    assert set(by_id) == {"youtube:UC1:aaa", "github:owner/repo:1"}
    assert by_id["youtube:UC1:aaa"].published == published
    assert by_id["github:owner/repo:1"].published == ""
    assert json.loads(by_id["youtube:UC1:aaa"].to_json())["published"] == published


# ── clause 2 — the writer holds what is older than a named window ────────────

def test_the_window_is_a_named_module_constant_of_at_least_30_days():
    """The acceptance grep looks for the constant by name, so it must be one."""
    assert vw_mod.MAX_ITEM_AGE_DAYS >= 30


def test_write_all_holds_items_past_the_window_and_prints_count_and_oldest(
        redirect_paths, capsys):
    """One fresh video is written; two stale ones (45 d, 31 d) are held and counted.

    The 453-day-old entry that reached the vault on 2026-09-22 is the shape being
    refused here, and the report has to name the oldest age — a bare count cannot say
    whether the day lost one week-old item or a year-old one.
    """
    populate_state("seed:already-known")
    write_intel_day(redirect_paths, DAY, [
        _item("yt_fresh", published=_published_iso(10)),
        _item("yt_stale_45", published=_published_iso(45)),
        _item("yt_stale_31", published=_published_iso(31)),
    ])

    written = vw_mod.write_all_to_vault(DAY)
    out = capsys.readouterr().out

    assert written == 1
    digest = redirect_paths / "obsidian" / "knowledge" / "ai-llms" / "youtube-digest.md"
    assert digest.read_text().count("\n## ") == 1
    assert "yt_stale" not in digest.read_text()
    assert "Held 2 item(s)" in out
    assert "oldest 45" in out
    assert "MAX_ITEM_AGE_DAYS" in out


def test_the_window_boundary_is_exclusive_at_the_declared_constant(redirect_paths,
                                                                   capsys):
    """Exactly `MAX_ITEM_AGE_DAYS` old is inside; one day past it is held."""
    populate_state("seed:already-known")
    inside = vw_mod.MAX_ITEM_AGE_DAYS
    write_intel_day(redirect_paths, DAY, [
        _item("yt_at_edge", published=_published_iso(inside)),
        _item("yt_one_past", published=_published_iso(inside + 1)),
    ])

    written = vw_mod.write_all_to_vault(DAY)

    assert written == 1
    assert "Held 1 item(s)" in capsys.readouterr().out


def test_write_item_to_vault_refuses_a_stale_item_on_its_own_surface(redirect_paths,
                                                                     capsys):
    """The other route into the vault, with no batch filter in front of it.

    `write_item_to_vault` is public; a guard on only `write_all_to_vault` is a guard
    on one of two write surfaces.
    """
    populate_state("seed:already-known")
    write_intel_day(redirect_paths, today_str(), [_item("yt_fresh_ctx",
                                                        published=_published_iso(1))])
    stale = _item("yt_stale_direct", published=_published_iso(453))

    assert vw_mod.write_item_to_vault(stale, profile_mod.load_profile()) is False

    out = capsys.readouterr().out
    assert "453" in out
    assert list((redirect_paths / "obsidian" / "knowledge").rglob("*.md")) == []


# ── clause 3 — a dateless source is inside the window ────────────────────────

def test_dateless_items_are_all_written_so_a_source_without_dates_is_never_zeroed(
        redirect_paths, capsys):
    """A GitHub row and a feed that omitted `published` both clear the gate.

    The whole day here is dateless, which is the case a naive age gate zeroes: if a
    source stops serving dates, that must read as a source without dates, never as
    "nothing to publish".
    """
    populate_state("seed:already-known")
    write_intel_day(redirect_paths, DAY, [
        _item("gh_dateless_1", source="github",
              title="vllm 0.29.1 released", summary="vllm", url="https://github.com/o/r"),
        _item("gh_dateless_2", source="github",
              title="vllm agent docs", summary="vllm", url="https://github.com/o/r2"),
        _item("yt_no_date", source="youtube", published=""),
    ])

    written = vw_mod.write_all_to_vault(DAY)
    out = capsys.readouterr().out

    assert written == 3
    assert "Held 0" not in out
    assert "Held" not in out


def test_an_unparseable_publish_date_is_treated_as_dateless(redirect_paths):
    """A malformed date must not become a 0-age or a held item by accident."""
    assert vw_mod.item_age_days(_item("yt_junk", published="not-a-date")) is None
    assert vw_mod.too_old_for_vault(_item("yt_junk", published="not-a-date")) is False
    assert vw_mod.item_age_days(_item("yt_blank", published="   ")) is None
    age = vw_mod.item_age_days(_item("yt_ok", published=_published_iso(45)))
    assert age == 45
    assert vw_mod.too_old_for_vault(_item("yt_ok", published=_published_iso(45))) is True


# ── clause 4 — an emptied dedup state blocks the writer ──────────────────────

# The measured shape of the state-loss day: 258 scored items (1007 raw). A normal day
# scores 9. The declared floor sits between them, so a real big day still writes.
STATE_LOSS_DAY_ITEMS = 258
NORMAL_DAY_ITEMS = 9


def _big_day():
    return [_item(f"yt_big_{i}", title="vllm agents", relevance=6,
                  published=_published_iso(1)) for i in range(STATE_LOSS_DAY_ITEMS)]


def test_an_empty_state_and_an_oversized_day_write_nothing(redirect_paths, capsys):
    """No `scanner-state.json` at all: report state loss, publish nothing.

    The 2026-09-22 run did exactly this and shipped 102 videos — 28 of them over a
    month old — because an empty `seen` list reads as "nothing has ever been seen",
    which makes the entire backlog look like today's news.
    """
    assert not state_mod.STATE_FILE.exists()
    write_intel_day(redirect_paths, DAY, _big_day())

    written = vw_mod.write_all_to_vault(DAY)
    out = capsys.readouterr().out

    assert written == 0
    assert "STATE LOSS" in out
    assert f"{STATE_LOSS_DAY_ITEMS} scored items" in out
    assert str(vw_mod.STATE_LOSS_MIN_SCORED_ITEMS) in out
    assert list((redirect_paths / "obsidian" / "knowledge").rglob("*.md")) == []
    # Nothing was claimed as written either: a partial state write would make the
    # next run believe the day had been published.
    assert not vw_mod.VAULT_WRITTEN_STATE.exists()


def test_a_populated_state_writes_the_same_oversized_day_normally(redirect_paths,
                                                                  capsys):
    """The one-variable difference is the state file, so it must be the whole cause."""
    populate_state("seed:already-known")
    write_intel_day(redirect_paths, DAY, _big_day())

    written = vw_mod.write_all_to_vault(DAY)
    out = capsys.readouterr().out

    assert written == STATE_LOSS_DAY_ITEMS
    assert "STATE LOSS" not in out
    digest = redirect_paths / "obsidian" / "knowledge" / "ai-llms" / "youtube-digest.md"
    assert digest.read_text().count("\n## ") == STATE_LOSS_DAY_ITEMS


def test_an_empty_state_below_the_floor_is_a_normal_day(redirect_paths, capsys):
    """9 scored items with no state is a quiet day, not state loss.

    Without this the guard would turn the absence of state into a permanent
    zero-write day, which is the opposite failure and just as silent.
    """
    write_intel_day(redirect_paths, DAY, [
        _item(f"yt_normal_{i}", published=_published_iso(1))
        for i in range(NORMAL_DAY_ITEMS)])

    written = vw_mod.write_all_to_vault(DAY)
    out = capsys.readouterr().out

    assert written == NORMAL_DAY_ITEMS
    assert "STATE LOSS" not in out


def test_an_empty_seen_list_in_an_existing_state_file_is_state_loss_too(
        redirect_paths, capsys):
    """`{"seen": []}` on disk behaves exactly like an absent file.

    Half 2's mechanism was a truncated state file, not a missing one — a file that
    exists and holds nothing is the shape the 2026-09-22 recovery actually left.
    """
    state_mod.STATE_FILE.write_text(json.dumps({"seen": [], "last_run": None}))
    write_intel_day(redirect_paths, DAY, _big_day())

    written = vw_mod.write_all_to_vault(DAY)

    assert written == 0
    assert "STATE LOSS" in capsys.readouterr().out


def test_write_item_to_vault_also_refuses_on_an_empty_state_and_an_oversized_day(
        redirect_paths, capsys):
    """Same guard on the single-item surface, reading the day's own scored file.

    No date parameter reaches this route, so the day it counts is the day of the call.
    """
    write_intel_day(redirect_paths, today_str(), _big_day())
    item = _item("yt_big_0", published=_published_iso(1))

    assert vw_mod.write_item_to_vault(item, profile_mod.load_profile()) is False

    assert "STATE LOSS" in capsys.readouterr().out
    assert list((redirect_paths / "obsidian" / "knowledge").rglob("*.md")) == []


# ── the process boundary autonomy task #30 actually runs ─────────────────────

def _cli_home(tmp_path):
    """A scratch HOME + LLOYD_DATA: the vault and the dedup state a real run resolves."""
    home = tmp_path / "home"
    feeds = home / "lloyd-data" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    (home / "obsidian").mkdir(parents=True)
    (home / "obsidian" / "interests.md").write_text(PROFILE_MD)
    return home, feeds


def _run_cli_write(home):
    env = dict(os.environ, HOME=str(home), LLOYD_DATA=str(home / "lloyd-data"),
               PYTHONPATH=f"{REPO_ROOT}:{INTEL_DIR}", INTEL_DISABLE_LLM="1")
    return subprocess.run(
        [sys.executable, "-m", "intel_pipeline", "--write"],
        cwd=str(INTEL_DIR), env=env, capture_output=True, text=True, timeout=180)


def _cli_day_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def test_cli_write_stage_blocks_on_state_loss_then_writes_across_the_process_boundary(
        tmp_path):
    """`python -m intel_pipeline --write` in its own process, over the files on disk.

    The in-process tests rebind module names; this one does not, so it is the test
    that proves the guard fires in the process the autonomy worker spawns — reading
    `intel-<date>.jsonl` and `scanner-state.json` from `LLOYD_DATA` and writing into
    the `HOME` vault. Same day, same rows: only the state file changes.
    """
    home, feeds = _cli_home(tmp_path)
    day = _cli_day_str()
    rows = "".join(_item(f"yt_cli_{i}", published=_published_iso(1)).to_json() + "\n"
                   for i in range(STATE_LOSS_DAY_ITEMS))
    (feeds / f"intel-{day}.jsonl").write_text(rows)

    blocked = _run_cli_write(home)
    assert blocked.returncode == 0, blocked.stderr[-2000:]
    assert "STATE LOSS" in blocked.stdout
    assert list((home / "obsidian" / "knowledge").rglob("*.md")) == []

    (feeds / "scanner-state.json").write_text(
        json.dumps({"seen": ["seed:already-known"], "last_run": None}))
    allowed = _run_cli_write(home)
    assert allowed.returncode == 0, allowed.stderr[-2000:]
    assert "STATE LOSS" not in allowed.stdout
    written = sorted(p.name for p in (home / "obsidian" / "knowledge").rglob("*.md"))
    assert written == ["youtube-digest.md"], (allowed.stdout[-1500:], written)
    digest = (home / "obsidian" / "knowledge" / "ai-llms" / "youtube-digest.md")
    assert digest.read_text().count("\n## ") == STATE_LOSS_DAY_ITEMS
