"""youtube_channel_monitor — the deterministic half of the channel digest.

`scripts/ai-engineer-monitor.py` grew a second channel and a session path on
2026-09-08. What is pinned here is what the rewrite must not have changed
and what the new path relies on:

  * the AI Engineer channel's state file, vault directory and front-matter
    `channel:` value are byte-for-byte what 600 existing notes and autonomy
    task #75 expect, and the old script name still works;
  * a channel tracked from a date has a *floor* and the new-video walk stops
    there — otherwise a 15-minute tick crawls the back catalogue forever;
  * the bundle a session reads is complete, the transcript is line-wrapped
    (the Read tool pages by line), and an existing note is reused in place;
  * state transitions the worker drives (`fetched` → `completed`/`failed`)
    and what `--pending` offers, including a bundle whose session died.

Module globals point at the live state and vault, so the autouse fixture
redirects every one of them; nothing here touches production files.
"""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "youtube_channel_monitor.py"
SHIM = ROOT / "scripts" / "ai-engineer-monitor.py"

_spec = importlib.util.spec_from_file_location("youtube_channel_monitor", SCRIPT)
M = importlib.util.module_from_spec(_spec)
sys.modules["youtube_channel_monitor"] = M
_spec.loader.exec_module(M)


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Every path the module writes to, redirected. `configure` rebinds the
    per-channel ones, so it is wrapped to re-apply the redirect."""
    real_configure = M.configure

    def configure(key):
        ch = real_configure(key)
        monkeypatch.setattr(M, "STATE_DIR", str(tmp_path / "state" / key))
        monkeypatch.setattr(M, "STATE_FILE", str(tmp_path / "state" / key / "seen.json"))
        monkeypatch.setattr(M, "VAULT_YT_DIR", str(tmp_path / "vault" / key))
        M._NOTE_INDEX.clear()
        return ch

    monkeypatch.setattr(M, "configure", configure)
    monkeypatch.setattr(M, "VAULT_GH_DIR", str(tmp_path / "vault" / "github"))
    monkeypatch.setattr(M, "VAULT_PAPER_DIR", str(tmp_path / "vault" / "papers"))
    monkeypatch.setattr(M, "TMP_CLONES", str(tmp_path / "clones"))
    monkeypatch.setattr(M, "REPORT_DIR", str(tmp_path / "vault" / "channel-eval"))
    configure("discover-ai")
    yield
    real_configure(M.DEFAULT_CHANNEL)
    M._NOTE_INDEX.clear()


def _state(**seen):
    return {"seen": dict(seen), "backfill_complete_through": None}


# ---------------------------------------------------------------------------
# Note filenames Obsidian Sync will store
# ---------------------------------------------------------------------------
#
# On 2026-09-10 twenty AI Engineer notes had a no-break space in their names
# (titles like "Composer\u00a0– Lee Robinson"). Obsidian Sync uploads such a
# file under the name with an ordinary space, so the local file never matches
# its own remote copy: the client re-detected them as new on every start, and
# every other device got a differently named file. A first reading of the
# upload log called them refused; that was a matching artifact — the log
# prints the ordinary space too.

_SPACES = ["\u00a0", "\u202f", "\u2009", "\u3000", "\t", "\n"]


@pytest.mark.parametrize("space", _SPACES)
def test_every_kind_of_space_becomes_a_dash(space):
    title = f"Building Cursor Composer{space}– Lee Robinson,{space}Cursor"
    assert M.slugify(title) == "building-cursor-composer-lee-robinson-cursor"


def test_the_cut_never_leaves_whitespace_or_a_trailing_dash():
    """The old cut came before the collapse, so an 80-character limit landing
    on a no-break space left the name ending in one — the note that ended
    `...voice-response-\\u00a0.md`."""
    slug = M.slugify("a" * 79 + "\u00a0" + "b" * 10)
    assert slug == "a" * 79
    for title in ("x" * 78 + " \u00a0 y", "Title — with — dashes —\u00a0", "  lead and trail  "):
        s = M.slugify(title)
        assert not any(ch.isspace() for ch in s)
        assert not s.startswith("-") and not s.endswith("-")


def test_slugify_is_idempotent():
    for title in ("Building Cursor Composer\u00a0– Lee Robinson, Cursor",
                  "AI Didn’t Kill the Web, It Moved in! — Olivier Leplus (AWS)",
                  "Café Déjà Vu"):
        once = M.slugify(title)
        assert M.slugify(once) == once


def test_accents_are_kept_and_composed():
    """They sync (measured), and composing them means macOS and Linux write
    the same bytes for the same name."""
    assert M.slugify("Café Déjà Vu") == "café-déjà-vu"


def test_reserved_and_zero_width_characters_are_dropped():
    assert M.slugify('a:b*c?"d<e>f|g\\h/i\u200bj k') == "abcdefghij-k"


def test_an_unsluggable_title_falls_back_to_the_video_id():
    assert M.target_note_path("🔥🔥🔥", "20260910", video_id="abc123").endswith(
        "20260910-video-abc123.md")
    assert M.target_note_path("OpenRAG: An open-source stack", "20260408").endswith(
        "20260408-openrag-an-open-source-stack.md")


@pytest.mark.parametrize("bad, fixed", [
    ("building-cursor-composer-\u00a0lee-robinson-cursor",
     "building-cursor-composer-lee-robinson-cursor"),
    ("voicevision-rag-integrating-visual-document-intelligence-with-voice-response-\u00a0",
     "voicevision-rag-integrating-visual-document-intelligence-with-voice-response"),
])
def test_the_twenty_real_names_come_out_clean(bad, fixed):
    """Two of the twenty, as they were written. Renaming them applies this
    same function to the part after the date."""
    assert M.slugify(bad, max_len=len(bad)) == fixed


# ---------------------------------------------------------------------------
# The registry and the shim
# ---------------------------------------------------------------------------


def test_ai_engineer_keeps_its_legacy_paths_and_handle():
    """600 notes carry `channel: aiDotEngineer`, and task #75's state lives at
    the old path. The rewrite may add channels, not move this one."""
    ch = M.CHANNELS["ai-engineer"]
    assert ch["channel_id"] == "UCLKPca3kwwd-B59HNr-_lvA"
    assert ch["handle"] == "aiDotEngineer"
    assert ch["state_dir"] == "~/.local/share/ai-engineer"
    assert ch["vault_dir"] == "~/obsidian/knowledge/youtube/AI_Engineer"
    assert M.DEFAULT_CHANNEL == "ai-engineer"


def test_discover_ai_is_registered_under_its_own_paths():
    ch = M.CHANNELS["discover-ai"]
    assert ch["channel_id"] == "UCfOvNb3xj28SNqPQ_JIbumg"
    assert ch["handle"] == "code4AI"
    assert ch["state_dir"] != M.CHANNELS["ai-engineer"]["state_dir"]
    assert ch["vault_dir"] != M.CHANNELS["ai-engineer"]["vault_dir"]


def test_configure_repoints_every_channel_global():
    M.configure("discover-ai")
    assert M.CHANNEL_KEY == "discover-ai"
    assert M.CHANNEL_HANDLE == "code4AI" and M.CHANNEL_NAME == "Discover AI"
    assert M.UPLOADS_PLAYLIST.endswith("list=UUfOvNb3xj28SNqPQ_JIbumg")
    assert M.channel_info()["key"] == "discover-ai"
    with pytest.raises(KeyError):
        M.configure("no-such-channel")


def test_the_old_script_name_still_runs_and_selects_ai_engineer():
    """Task #75 and its skill call `ai-engineer-monitor.py`; the vault is not
    in this repo, so the name must keep working. `--help` exits before any
    network call, which is what makes this cheap."""
    r = subprocess.run([sys.executable, str(SHIM), "--help"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "--channel" in r.stdout
    assert '"--channel", "ai-engineer"' in SHIM.read_text()


def test_note_prompt_names_the_active_channel(monkeypatch):
    """The note template used to hard-code `channel: aiDotEngineer` in two
    places. A Discover AI note written through the script path must not
    claim to be from AI Engineer."""
    seen = {}
    monkeypatch.setattr(M, "call_llm", lambda system, user, **kw: seen.setdefault("prompt", user) and None)
    M.generate_knowledge_note("vid1", "A talk", "20260901", "desc", "transcript words", M.extract_entities(""))
    assert "channel: code4AI" in seen["prompt"]
    assert "@code4AI (Discover AI)" in seen["prompt"]
    assert "aiDotEngineer" not in seen["prompt"]


# ---------------------------------------------------------------------------
# The window and the floor
# ---------------------------------------------------------------------------


def _v(i, vid=None):
    return {"id": vid or f"v{i}", "title": f"Video {i}", "published": ""}


def test_select_since_stops_after_two_older_and_records_the_floor():
    dated = [(_v(1), "20260908"), (_v(2), "20260901"), (_v(3), ""),        # undated: inside
             (_v(4), "20260701"), (_v(5), "20260820"),                     # one older is not the end
             (_v(6), "20260601"), (_v(7), "20260515"), (_v(8), "20260907")]  # two older: stop
    inside, floor = M.select_since(dated, "20260710")
    assert [v["id"] for v, _ in inside] == ["v1", "v2", "v3", "v5"]
    assert floor == "v5"
    assert M.select_since([], "20260710") == ([], None)


def test_videos_above_floor_bounds_the_walk():
    videos = [_v(i) for i in range(1, 7)]
    assert M.videos_above_floor(_state(), videos) == videos, "no floor: the old behaviour"
    st = _state(); st["floor_video_id"] = "v3"
    assert [v["id"] for v in M.videos_above_floor(st, videos)] == ["v1", "v2", "v3"]
    # The floor video was deleted from the channel: stop after the last seen.
    st = _state(v2={"status": "completed"}, v4={"status": "completed"})
    st["floor_video_id"] = "gone"
    assert [v["id"] for v in M.videos_above_floor(st, videos)] == ["v1", "v2", "v3", "v4"]


def test_get_next_video_never_goes_below_the_floor(monkeypatch):
    monkeypatch.setattr(M, "is_video_playable", lambda vid: (True, "", "20260908"))
    videos = [_v(1), _v(2), _v(3)]
    st = _state(v1={"status": "completed"}, v2={"status": "completed"})
    st["floor_video_id"] = "v2"
    assert M.get_next_video(st, videos) is None
    del st["floor_video_id"]
    assert M.get_next_video(st, videos)["id"] == "v3"


def test_register_since_uses_existing_note_dates_and_requeues(monkeypatch, tmp_path):
    """AI Engineer's state rows all carry `published: ""` while their notes
    carry the real date, and probing yt-dlp for 300 dates is 15 minutes.
    With --requeue a completed video in the window goes back to pending
    keeping its note, so the session refreshes it in place."""
    M.configure("ai-engineer")
    vault = Path(M.VAULT_YT_DIR); vault.mkdir(parents=True)
    (vault / "20260901-old-talk.md").write_text(
        "---\nsegment: knowledge\nvideo_id: v1\nchannel: aiDotEngineer\npublished: 20260901\n---\n# Old talk\n")
    probed = []
    monkeypatch.setattr(M, "_upload_date_of", lambda vid: probed.append(vid) or {"v2": "20260905", "v3": "20260101"}.get(vid, ""))
    monkeypatch.setattr(M, "datetime", _FrozenDatetime)
    st = _state(v1={"status": "completed", "title": "Old talk", "published": "",
                    "youtube_note": "/nonexistent/video-v1.md"})
    result = M.register_since(st, [_v(1), _v(2), _v(3), _v(4)], 60, requeue=True)
    assert "v1" not in probed, "the note's front matter already had the date"
    assert st["seen"]["v1"]["status"] == "pending"
    assert st["seen"]["v1"]["existing_note"] == str(vault / "20260901-old-talk.md")
    assert st["seen"]["v1"]["published"] == "20260901"
    assert st["seen"]["v2"]["status"] == "pending"
    assert "v3" not in st["seen"] and "v4" not in st["seen"], "below the window"
    assert st["floor_video_id"] == "v2" and result["requeued"] == 1 and result["added"] == 1


def test_register_new_dates_a_new_upload_from_the_playability_probe(monkeypatch):
    """The flat playlist listing carries no dates, so `register_new` used to
    write `published: ""` — and `pending_entries` sorts newest-first, which
    puts "" *last*. Every genuinely new upload therefore went to the back of
    the queue that exists to reach new uploads first. Harmless on an empty
    board; on 2026-09-09 AI Engineer had 170 pending, so a video published
    that morning would have waited behind all of them.

    The date is free: `is_video_playable` already does a full extraction.
    """
    monkeypatch.setattr(M, "is_video_playable",
                        lambda vid: (True, "", {"v3": "20260909"}.get(vid, "")))
    st = _state(v1={"status": "completed", "published": "20260801"},
                v2={"status": "pending", "published": "20260715"})

    added = M.register_new(st, [_v(3), _v(1), _v(2)])

    assert added == ["v3"]
    assert st["seen"]["v3"]["published"] == "20260909"
    assert [r["video_id"] for r in M.pending_entries(st)] == ["v3", "v2"], \
        "the new upload sorts ahead of the backlog, which is the whole point"


def test_a_probe_that_cannot_date_a_video_still_registers_it(monkeypatch):
    """An unresolvable date is "" — exactly the old behaviour. A video with
    no date is worth less queue position than it is worth losing."""
    monkeypatch.setattr(M, "is_video_playable", lambda vid: (True, "", ""))
    st = _state(v1={"status": "completed", "published": "20260801"})
    assert M.register_new(st, [_v(2), _v(1)]) == ["v2"]
    assert st["seen"]["v2"]["published"] == ""
    assert [r["video_id"] for r in M.pending_entries(st)] == ["v2"]


def test_the_playability_probe_reports_the_date_it_already_fetched(monkeypatch):
    """A three-tuple, not a second yt-dlp call: the probe is a full extract
    and `upload_date` is in the payload it already parses."""
    payload = json.dumps({"playable": True, "status": "", "availability": "public",
                          "reason": "", "play_reason": "", "upload_date": "20260909"})
    monkeypatch.setattr(M.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, payload, ""))
    assert M.is_video_playable("v1") == (True, "", "20260909")

    premiere = json.dumps({"playable": False, "status": "premiere_scheduled",
                           "availability": "", "reason": "Premieres in 2 hours",
                           "play_reason": "", "upload_date": "20260910"})
    monkeypatch.setattr(M.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, premiere, ""))
    playable, reason, date = M.is_video_playable("v2")
    assert playable is False and "Premieres" in reason and date == "20260910"

    monkeypatch.setattr(M.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "boom"))
    assert M.is_video_playable("v3") == (False, "yt-dlp unavailable", "")


class _FrozenDatetime(M.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 8, 12, 0, tzinfo=tz)


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


def test_wrap_transcript_makes_a_pageable_file():
    one_line = " ".join(["word"] * 5000)
    wrapped = M.wrap_transcript(one_line)
    assert max(len(l) for l in wrapped.splitlines()) <= M.TRANSCRIPT_WRAP
    assert wrapped.split() == one_line.split(), "wrapping may not change the words"
    assert wrapped.count("\n") > 200


def _stub_fetchers(monkeypatch, transcript="the talk " * 400, metadata=None):
    monkeypatch.setattr(M, "_fetch_text", lambda url, timeout=6: (_ for _ in ()).throw(OSError("no network in tests")))
    monkeypatch.setattr(M, "fetch_video_metadata", lambda vid: metadata if metadata is not None else {
        "title": "Adaptive Harness", "upload_date": "20260901", "channel": "Discover AI",
        "description": "see https://github.com/acme/harness and arxiv.org/abs/2608.13560"})
    monkeypatch.setattr(M, "fetch_transcript", lambda vid: (transcript, None))
    monkeypatch.setattr(M, "clone_and_note", lambda url: {"owner": "acme", "repo": "harness", "note_path": "/v/github/acme-harness.md"})
    monkeypatch.setattr(M, "fetch_arxiv_paper", lambda a: None)
    monkeypatch.setattr(M, "fetch_generic_paper", lambda u: None)


def test_build_bundle_writes_what_the_session_reads(monkeypatch):
    _stub_fetchers(monkeypatch, transcript="see https://github.com/acme/harness " + "the talk " * 400)
    meta, err = M.build_bundle("abc123", "placeholder", "")
    assert err is None
    bdir = Path(meta["bundle_dir"])
    assert bdir == Path(M.STATE_DIR) / "bundles" / "abc123"
    assert (bdir / "transcript.txt").is_file() and (bdir / "meta.json").is_file()
    assert json.loads((bdir / "meta.json").read_text())["video_id"] == "abc123"
    assert meta["title"] == "Adaptive Harness" and meta["published"] == "20260901"
    assert meta["channel_handle"] == "code4AI" and meta["channel_key"] == "discover-ai"
    assert meta["existing_note"] is None
    assert meta["target_note"] == str(Path(M.VAULT_YT_DIR) / "20260901-adaptive-harness.md")
    assert meta["enrichment"]["github"][0]["note_path"].endswith("acme-harness.md")
    assert meta["transcript_lines"] > 20 and meta["transcript_words"] > 400
    assert M.load_bundle("abc123")["video_id"] == "abc123"
    assert M.load_bundle("nope") is None
    # The snapshot is always written, even with every source unreachable.
    m = json.loads(Path(meta["measurements_path"]).read_text())
    assert Path(meta["measurements_path"]).parent == bdir
    assert len(m["errors"]) >= 2 and "unavailable" in meta["measurements_summary"]


def test_measurements_capture_parses_the_counters_the_eval_needs(monkeypatch, tmp_path):
    """`http_fetch` refuses loopback, so the script snapshots the numbers into
    the bundle. The prefix-cache rate is the one the first review turned on."""
    prom = "\n".join([
        "# HELP vllm:prefix_cache_queries_total x",
        'vllm:prefix_cache_queries_total{model_name="primary"} 1000',
        'vllm:prefix_cache_hits_total{model_name="primary"} 687',
        'vllm:kv_cache_usage_perc{model_name="primary"} 0.79',
        'vllm:num_requests_running{model_name="primary"} 5',
        'vllm:other_metric{model_name="primary"} 42',
    ])
    dash = json.dumps({"vllm": {"engines": []}, "workers": {"slots": 2}, "primary": {"big": "x" * 10}})
    monkeypatch.setattr(M, "_fetch_text", lambda url, timeout=6: prom if url.endswith("/metrics") else dash)
    import os, time
    base = tmp_path / "baselines"; base.mkdir()
    old = base / "nightly-20260908-1.json"; old.write_text(json.dumps({"measured_at": "t1", "overall": {"entity_hit_rate": 0.5, "doc_hit_rate": 0.9}}))
    # The runner's real shape: metrics under `summary`, time under `ran_at`.
    new = base / "nightly-20260909-1.json"; new.write_text(json.dumps({"ran_at": "t2", "summary": {"overall": {"entity_hit_rate": 0.55, "doc_hit_rate": 0.95}, "by_category": {}}}))
    # Newer by mtime but no metrics: a rebuild-after file is a corpus description.
    corpus = base / "rebuild-after-20260909-2.json"; corpus.write_text(json.dumps({"label": "rebuild", "ran_at": "t3", "corpus": {"docs": 1}}))
    now = time.time()
    os.utime(old, (now - 7200, now - 7200)); os.utime(new, (now - 3600, now - 3600)); os.utime(corpus, (now, now))
    monkeypatch.setattr(M, "EVAL_BASELINES_DIR", str(base))
    bdir = tmp_path / "b"; bdir.mkdir()
    path, m = M.capture_measurements(str(bdir))
    assert Path(path) == bdir / "measurements.json"
    assert m["vllm"]["prefix_cache_hit_rate_since_boot"] == 0.687
    assert m["vllm"]["vllm:num_requests_running"] == 5 and "vllm:other_metric" not in m["vllm"]
    assert set(m["dashboard"]) == {"vllm", "workers"}, "only the sections a verdict needs"
    assert m["retrieval_eval"]["measured_at"] == "t2" and m["errors"] == [], "newest *with metrics*, not newest file"
    # A nightly within a day of the newest run is preferred over a newer
    # ad-hoc check (it is the number the dashboard and the gate quote); once
    # no nightly is that fresh, the newest run with metrics wins. Empty: None.
    check = base / "automod-check-20260909-3.json"; check.write_text(json.dumps({"label": "check", "ran_at": "t4", "overall": {"entity_hit_rate": 0.6}}))
    os.utime(new, (now - 200000, now - 200000))
    assert M.newest_retrieval_baseline(str(base))["measured_at"] == "t1", "day-old nightly beats the newer check"
    os.utime(old, (now - 200000, now - 200000))
    assert M.newest_retrieval_baseline(str(base))["measured_at"] == "t4"
    assert M.newest_retrieval_baseline(str(tmp_path / "empty")) is None
    summary = M.measurements_summary(m)
    assert "68.7%" in summary and "79%" in summary and "entity_hit_rate 0.55" in summary


def test_build_bundle_reuses_an_existing_note_path(monkeypatch):
    _stub_fetchers(monkeypatch)
    vault = Path(M.VAULT_YT_DIR); vault.mkdir(parents=True)
    old = vault / "20260901-some-other-slug.md"
    old.write_text("---\nvideo_id: abc123\npublished: 20260901\n---\n# x\n")
    meta, _ = M.build_bundle("abc123")
    assert meta["existing_note"] == str(old)
    assert meta["target_note"] == str(old), "refresh in place, never a second note"


def test_build_bundle_fails_only_on_the_transcript(monkeypatch):
    _stub_fetchers(monkeypatch, metadata={})
    monkeypatch.setattr(M, "fetch_video_metadata", lambda vid: (_ for _ in ()).throw(RuntimeError("yt-dlp down")))
    meta, err = M.build_bundle("abc123", "Given title", "20260801")
    assert err is None and meta["title"] == "Given title" and meta["published"] == "20260801"
    monkeypatch.setattr(M, "fetch_transcript", lambda vid: (None, "NoTranscriptFound"))
    meta, err = M.build_bundle("abc123")
    assert meta is None and "transcript" in err.lower()


# ---------------------------------------------------------------------------
# State transitions and what the worker is offered
# ---------------------------------------------------------------------------


def test_worker_state_transitions_and_pending_order():
    st = _state(
        new={"status": "pending", "title": "New", "published": "20260907"},
        stale_bundle={"status": "fetched", "title": "Died mid-session", "published": "20260908"},
        done={"status": "completed", "title": "Done", "published": "20260906"},
        dead={"status": "failed", "title": "Dead", "published": "20260905",
              "failure_count": 12, "transcript_error": "no transcript"},
        undated={"status": "pending", "title": "Undated", "published": ""},
    )
    rows = M.pending_entries(st)
    assert [r["video_id"] for r in rows] == ["stale_bundle", "new", "undated"], \
        "newest first, unknown dates last, exhausted failures excluded"

    meta = {"title": "New", "published": "20260907", "bundle_dir": "/b/new",
            "fetched_at": "t", "existing_note": None}
    M.mark_fetched(st, "new", meta)
    assert st["seen"]["new"]["status"] == "fetched"
    M.mark_failed(st, "new", "turn ended (max_turns) without a note")
    e = st["seen"]["new"]
    assert e["status"] == "failed" and e["failure_count"] == 1 and "transcript_error" not in e
    M.mark_failed(st, "new", "Could not fetch transcript: NoTranscriptFound")
    assert st["seen"]["new"]["transcript_error"].startswith("Could not fetch transcript")
    M.mark_completed(st, "new", "/v/20260907-new.md", {"relevance": 80, "verdict": "actionable"})
    e = st["seen"]["new"]
    assert e["status"] == "completed" and e["youtube_note"] == "/v/20260907-new.md"
    assert e["eval"]["verdict"] == "actionable"
    assert not {"failure_count", "last_attempt_at", "transcript_error", "last_error"} & set(e)


def test_register_new_registers_everything_above_the_floor(monkeypatch):
    monkeypatch.setattr(M, "is_video_playable",
                        lambda vid: (False, "Premieres in 2 hours", "") if vid == "v2" else (True, "", "20260908"))
    st = _state(v4={"status": "completed"}); st["floor_video_id"] = "v4"
    added = M.register_new(st, [_v(1), _v(2), _v(3), _v(4), _v(5)])
    assert added == ["v1", "v3"]
    assert st["seen"]["v2"]["status"] == "deferred"
    assert "v5" not in st["seen"], "below the floor"
    assert Path(M.STATE_FILE).is_file(), "register_new persists"


# ---------------------------------------------------------------------------
# The report and the CLI's JSON line
# ---------------------------------------------------------------------------


def test_eval_report_is_a_projection_of_state():
    st = _state(
        a={"status": "completed", "title": "Best | talk", "published": "20260901",
           "eval": {"relevance": 85, "verdict": "actionable", "idea": "Adopt X", "filed": 523}},
        b={"status": "completed", "title": "Meh", "published": "20260902",
           "eval": {"relevance": 20, "verdict": "background", "idea": ""}},
        c={"status": "completed", "title": "No eval yet", "published": "20260903"},
    )
    path = Path(M.write_eval_report(st))
    assert path == Path(M.REPORT_DIR) / "discover-ai.md"
    text = path.read_text()
    fm = yaml.safe_load(text.split("---")[1])
    assert fm["segment"] == "projects" and "channel-eval" in fm["tags"]
    assert "**2 evaluated**" in text and "**1 backlog draft(s) filed.**" in text
    assert text.index("Best \\| talk") < text.index("| 20 | background"), "best first"
    assert "#523" in text and "No eval yet" not in text
    # Regeneration overwrites rather than appends.
    M.write_eval_report(st)
    assert path.read_text().count("## All verdicts") == 1


def _run_cli(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        sys_argv = sys.argv
        sys.argv = ["youtube_channel_monitor.py", *argv]
        try:
            M.main()
        finally:
            sys.argv = sys_argv
    out = buf.getvalue()
    last = [l for l in out.splitlines() if l.startswith(M.JSON_MARK)]
    return out, (json.loads(last[-1][len(M.JSON_MARK):]) if last else None)


def test_offline_cli_modes_never_list_the_channel(monkeypatch):
    """`--pending`, `--fetch`, `--complete`, `--fail` and `--eval-report` are
    the worker's calls; none may hit YouTube for the channel listing."""
    monkeypatch.setattr(M, "fetch_channel_videos",
                        lambda limit=200: (_ for _ in ()).throw(AssertionError("listed the channel")))
    _stub_fetchers(monkeypatch)
    Path(M.STATE_DIR).mkdir(parents=True)
    M.save_state(_state(abc123={"status": "pending", "title": "T", "published": "20260901"}))

    _, out = _run_cli(["--channel", "discover-ai", "--pending", "--json"])
    assert out["ok"] and out["pending"][0]["video_id"] == "abc123"

    _, out = _run_cli(["--channel", "discover-ai", "--fetch", "abc123", "--json"])
    assert out["ok"] and out["meta"]["target_note"].endswith("20260901-adaptive-harness.md")
    assert M.load_state()["seen"]["abc123"]["status"] == "fetched"

    text, out = _run_cli(["--channel", "discover-ai", "--fetch", "abc123", "--json"])
    assert "Reusing bundle" in text and out["ok"]

    _, out = _run_cli(["--channel", "discover-ai", "--complete", "abc123", "--note", "/v/n.md",
                       "--eval-json", json.dumps({"relevance": 75, "verdict": "actionable"}), "--json"])
    assert out["ok"]
    e = M.load_state()["seen"]["abc123"]
    assert e["status"] == "completed" and e["eval"]["relevance"] == 75

    _, out = _run_cli(["--channel", "discover-ai", "--fail", "abc123", "--reason", "boom", "--json"])
    assert out["ok"] and out["failure_count"] == 1

    _, out = _run_cli(["--channel", "discover-ai", "--eval-report", "--json"])
    assert out["ok"] and Path(out["report"]).is_file()


def test_fetch_records_a_transcript_failure_as_a_retryable_attempt(monkeypatch):
    _stub_fetchers(monkeypatch)
    monkeypatch.setattr(M, "fetch_transcript", lambda vid: (None, "network reset"))
    Path(M.STATE_DIR).mkdir(parents=True)
    M.save_state(_state(abc123={"status": "pending", "title": "T"}))
    _, out = _run_cli(["--channel", "discover-ai", "--fetch", "abc123", "--json"])
    assert out["ok"] is False and out["failure_count"] == 1
    e = M.load_state()["seen"]["abc123"]
    assert e["status"] == "failed" and M._is_retry_eligible(e, M.datetime.now(M.timezone.utc)) is False, \
        "the retry interval has not elapsed yet"
    e["last_attempt_at"] = "2020-01-01T00:00:00+00:00"
    assert M._is_retry_eligible(e, M.datetime.now(M.timezone.utc)) is True
