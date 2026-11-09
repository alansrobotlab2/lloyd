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
    monkeypatch.setattr(M, "PAPERS_STATE_DIR", str(tmp_path / "state" / "papers"))
    monkeypatch.setattr(M, "ARXIV_SPACING_SECONDS", 0.0)

    def no_network(params):
        raise AssertionError(f"a test reached the arXiv API: {params}")

    monkeypatch.setattr(M, "_arxiv_query", no_network)
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
    monkeypatch.setattr(M, "_arxiv_query", lambda params: [])
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


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------
#
# Before 2026-10-09 the paper path was one function fed by a regex over the
# transcript. Counted over the 517 bundles on disk that day: no transcript held
# an arXiv id or a paper URL, so no paper note was ever written, while 21
# descriptions named an id and most of Discover AI's named a title. What is
# pinned here is the three things that had to change — the description is
# scanned, a title resolves to a record only on a near-exact match, and a
# registered paper is a row the worker can take — and the two notes the
# registry must never overwrite.

RECORD = {
    "arxiv_id": "2609.04148",
    "title": "Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments",
    "summary": "We reconstruct executable terminal environments from recorded agent trajectories.",
    "authors": ["Jie Wu", "Zhenru Zhang"], "published": "2026-09-03", "categories": ["cs.AI"],
}
VIDEO = {"channel": "discover-ai", "channel_name": "Discover AI", "video_id": "abc123",
         "title": "Qwen Learns From Self-Evolving RL Worlds", "published": "20260920",
         "note": "/vault/discover-ai/20260920-qwen-learns.md"}


def _arxiv(monkeypatch, *records):
    """Answer the arXiv API from a fixed set of records, and record each call."""
    calls = []

    def query(params):
        calls.append(params)
        if "id_list" in params:
            return [r for r in records if r["arxiv_id"] == params["id_list"]]
        return list(records)

    monkeypatch.setattr(M, "_arxiv_query", query)
    return calls


def test_an_arxiv_id_is_checked_not_just_matched():
    assert M.normalize_arxiv_id("arXiv:2609.01481v1") == "2609.01481"
    assert M.normalize_arxiv_id("https://arxiv.org/pdf/1706.03762") == "1706.03762"
    # A year range, a price, a thirteenth month, and a four-digit number on a
    # 2020s id all fit \d{4}\.\d{4,5}.
    for not_an_id in ("2010.2015", "$1999.9999", "2613.00001", "2609.0148", "0703.1234"):
        assert M.normalize_arxiv_id(not_an_id) is None, not_an_id


def test_the_description_is_where_the_links_are():
    spoken = "today we look at a new paper from the qwen team on terminal environments"
    description = ("All rights w/ authors:\narXiv:2609.04148v2\nhttps://huggingface.co/papers/2607.21612\n"
                   "https://huggingface.co/Qwen/Qwen3\nhttps://openreview.net/forum?id=AbC_12.\n"
                   "https://arxiv.org/abs/2609.04148")
    assert M.extract_entities(spoken)["paper_arxiv"] == []
    got = M.extract_entities(spoken + "\n" + description)
    assert got["paper_arxiv"] == ["2609.04148", "2607.21612"], "deduplicated, version dropped"
    # arXiv is registered by id and a model page is not a paper: neither is a
    # generic paper URL, which is what wrote a second stub per arXiv link.
    assert got["paper_urls"] == ["https://openreview.net/forum?id=AbC_12"]


def test_build_bundle_registers_a_paper_the_description_names(monkeypatch):
    _stub_fetchers(monkeypatch, metadata={"title": "Self-Evolving RL Worlds", "upload_date": "20260920",
                                         "description": "All rights w/ authors.\narXiv:2609.04148"})
    calls = _arxiv(monkeypatch, RECORD)
    meta, err = M.build_bundle("abc123")
    assert err is None and calls == [{"id_list": "2609.04148"}]
    (paper,) = meta["enrichment"]["papers"]
    note = Path(paper["note_path"])
    assert note.name == "2609.04148-terminal-universe-turning-agent-trajectories-into-scalable-t.md"
    text = note.read_text()
    assert "arxiv_id: '2609.04148'" in text and "digest: abstract" in text
    assert yaml.safe_load(text.split("---")[1])["title"] == RECORD["title"], "a colon in a title is quoted"
    # The video's own note is linked back, at the path the session will write.
    assert f"- [[{Path(meta['target_note']).stem}]] — Discover AI, 20260920" in text
    assert text.rstrip().splitlines()[-3] == "## Discussed in", "the last section"
    assert M.paper_pending(M.load_papers())[0]["arxiv_id"] == "2609.04148"


def test_an_unreadable_registry_costs_the_papers_not_the_video(monkeypatch):
    _stub_fetchers(monkeypatch, metadata={"title": "T", "upload_date": "20260920",
                                         "description": "arXiv:2609.04148"})
    Path(M.PAPERS_STATE_DIR).mkdir(parents=True)
    Path(M.papers_state_file()).write_text("{not json")
    meta, err = M.build_bundle("abc123")
    assert err is None and meta["enrichment"]["papers"] == []
    assert Path(M.papers_state_file()).read_text() == "{not json", "never saved over"


def test_a_title_resolves_only_on_a_near_exact_match(monkeypatch):
    other = {**RECORD, "arxiv_id": "2608.11111",
             "title": "Terminal Environments for Agents: A Survey of Trajectories"}
    _arxiv(monkeypatch, other, RECORD)
    assert M.resolve_arxiv_title(RECORD["title"])["arxiv_id"] == "2609.04148"
    # Punctuation and case are the uploader's; the words are the paper's.
    assert M.resolve_arxiv_title(
        "terminal universe - turning agent trajectories into scalable terminal environments"
    )["arxiv_id"] == "2609.04148"
    # A title given without its subtitle, four words or more.
    _arxiv(monkeypatch, {**RECORD, "title": "Is Your Model Thinking or Just Stagnating? PUMA: Diagnosing Pathology"})
    assert M.resolve_arxiv_title("Is Your Model Thinking or Just Stagnating?") is not None
    # A wrong paper filed under a video is worse than none.
    _arxiv(monkeypatch, other)
    assert M.resolve_arxiv_title(RECORD["title"]) is None
    calls = _arxiv(monkeypatch, RECORD)
    assert M.resolve_arxiv_title("Terminal Universe") is None and calls == [], "two words is not a title"


def test_a_hand_written_note_is_never_queued_or_touched(monkeypatch):
    calls = _arxiv(monkeypatch, RECORD)
    papers = Path(M.VAULT_PAPER_DIR)
    papers.mkdir(parents=True)
    hand = papers / "2609.04148-terminal-universe.md"
    hand.write_text("---\ntype: notes\narxiv_id: '2609.04148'\ntitle: 'Terminal-Universe'\n---\n\n# Mine\n")
    before = hand.read_text()
    reg = M.load_papers()
    row = M.register_paper(reg, "2609.04148", VIDEO)
    assert row["status"] == "completed" and row["note_path"] == str(hand) and row["preexisting"]
    assert calls == [], "no lookup for a paper the vault already holds"
    assert hand.read_text() == before and M.paper_pending(reg) == []
    assert [p.name for p in papers.iterdir()] == [hand.name], "no second note beside it"


def test_a_second_video_is_added_to_the_placeholder_and_appended_to_a_full_note(monkeypatch):
    _arxiv(monkeypatch, RECORD)
    reg = M.load_papers()
    row = M.register_paper(reg, "2609.04148", VIDEO)
    second = {**VIDEO, "video_id": "def456", "published": "20260925", "note": "/v/20260925-part-two.md"}
    M.register_paper(reg, "2609.04148", second)
    M.register_paper(reg, "2609.04148", second)
    stub = Path(row["note_path"]).read_text()
    assert stub.count("[[20260925-part-two]]") == 1 and "[[20260920-qwen-learns]]" in stub
    assert len(row["videos"]) == 2

    # Once a session has written the full note, the file is the session's: a
    # third video is one appended line under the last section, nothing else.
    full = M.paper_note_header(row, M.DIGEST_FULL) + "\n## Summary\n\nBody.\n\n## Discussed in\n\n- [[a]]\n"
    Path(row["note_path"]).write_text(full)
    third = {**VIDEO, "video_id": "ghi789", "published": "20261001", "note": "/v/20261001-part-three.md"}
    M.register_paper(reg, "2609.04148", third)
    assert Path(row["note_path"]).read_text() == full + "- [[20261001-part-three]] — Discover AI, 20261001\n"


def test_a_failed_paper_waits_out_its_spacing_and_then_gives_up(monkeypatch):
    from datetime import datetime, timedelta, timezone
    _arxiv(monkeypatch, RECORD)
    reg = M.load_papers()
    row = M.register_paper(reg, "2609.04148", VIDEO)
    M.mark_paper_failed(row, "Could not fetch paper text: html: 404; pdf: timed out")
    now = datetime.now(timezone.utc)
    assert M.paper_pending(reg, now) == []
    later = now + timedelta(seconds=M.RETRY_INTERVAL_SECONDS + 1)
    assert [r["arxiv_id"] for r in M.paper_pending(reg, later)] == ["2609.04148"]
    row["failure_count"] = M.PAPER_MAX_ATTEMPTS
    assert M.paper_pending(reg, later) == []


LATEXML = """<html><head><title>x</title><style>.a{}</style></head><body>
<nav><a href="/">arXiv</a></nav>
<article class="ltx_document">
<h1 class="ltx_title">Terminal-Universe</h1>
<section><h2>1 Introduction</h2>
<p>We reach <math alttext="47.9\\%"><semantics><mn>47.9</mn><annotation>47.9\\%</annotation></semantics></math> on
the benchmark<br>and more.</p>
<table><tr><th>Model</th><th>Score</th></tr><tr><td>Ours</td><td>47.9</td></tr></table></section>
<section class="ltx_bibliography"><h2>References</h2><ul><li>Vaswani et al. Attention.</li></ul></section>
<section><h2>Appendix A</h2><p>Kept, because it follows the references.</p></section>
</article><footer>Report issue</footer></body></html>"""


def test_the_html_page_becomes_text_a_session_can_page():
    text, truncated = M.tidy_paper_text(M.html_to_text(LATEXML))
    assert not truncated
    assert "# Terminal-Universe" in text and "## 1 Introduction" in text and "## Appendix A" in text
    assert "47.9\\%" in text and text.count("47.9\\%") == 1, "a formula is its TeX source, once"
    assert "| Model | Score" in text and "| Ours | 47.9" in text
    for dropped in ("Vaswani", "References", "Report issue", "arXiv"):
        assert dropped not in text, dropped
    assert "Kept, because it follows the references." in text

    long, truncated = M.tidy_paper_text("word " * 200_000)
    assert truncated and len(long) <= M.PAPER_TEXT_MAX_CHARS + 200
    assert max(len(line) for line in long.splitlines()) <= 100
    assert long.splitlines()[-1].startswith("[cut here")


def test_the_paper_cli_round_trip(monkeypatch, tmp_path):
    """What the worker drives: register by title → pending → fetch → complete."""
    _arxiv(monkeypatch, RECORD)
    note = tmp_path / "20260920-qwen-learns.md"
    note.write_text("---\nvideo_id: abc123\n---\n")
    M.save_state(_state(abc123={"title": VIDEO["title"], "published": "20260920", "status": "completed",
                                "youtube_note": str(note)}))
    _, out = _run_cli(["--channel", "discover-ai", "--paper-register=abc123",
                    "--paper-refs=" + json.dumps([RECORD["title"], "A Title Nobody Wrote About Zebras Ever", "none"]),
                    "--json"])
    assert [r["arxiv_id"] for r in out["registered"]] == ["2609.04148"]
    assert out["unresolved"] == ["A Title Nobody Wrote About Zebras Ever"], "and `none` is not a reference"
    row = M.load_papers()["papers"]["2609.04148"]
    assert "[[20260920-qwen-learns]]" in Path(row["note_path"]).read_text(), \
        "a paper named only by the session is linked from its own note, since the video's is already written"

    _, out = _run_cli(["--papers-pending", "--json"])
    assert [r["arxiv_id"] for r in out["pending"]] == ["2609.04148"]

    monkeypatch.setattr(M, "fetch_paper_text", lambda arxiv_id: ("A paragraph. " * 400, "html"))
    _, out = _run_cli(["--paper-fetch=2609.04148", "--json"])
    meta = out["meta"]
    assert out["ok"] and Path(meta["text_path"]).read_text().startswith("A paragraph.")
    assert "digest: full" in meta["note_header"] and meta["text_source"] == "html"
    assert meta["target_note"].endswith(".md") and meta["discussed"] and meta["discussed"][0].startswith("- ")
    assert _run_cli(["--papers-pending", "--json"])[1]["pending"][0]["status"] == "fetched", \
        "a bundle whose session never reported back is offered again"

    assert _run_cli(["--paper-complete=2609.04148", "--json"])[1]["ok"]
    assert _run_cli(["--papers-pending", "--json"])[1]["pending"] == []
    assert _run_cli(["--paper-fetch=2601.00001", "--json"])[1]["ok"] is False, "not a registered paper"


def test_a_paper_with_no_text_is_a_counted_failure(monkeypatch):
    _arxiv(monkeypatch, RECORD)
    reg = M.load_papers()
    M.register_paper(reg, "2609.04148", VIDEO)
    M.save_papers(reg)
    monkeypatch.setattr(M, "fetch_paper_text", lambda arxiv_id: (None, "html: 404; pdf: timed out"))
    _, out = _run_cli(["--paper-fetch=2609.04148", "--json"])
    assert out["ok"] is False and out["failure_count"] == 1 and "paper text" in out["error"]
    assert M.load_papers()["papers"]["2609.04148"]["status"] == "failed"


def test_the_backfill_reads_bundles_on_disk_and_only_completed_videos(monkeypatch, tmp_path):
    calls = _arxiv(monkeypatch, RECORD)
    note = tmp_path / "20260901-done.md"
    note.write_text("---\nvideo_id: done1\n---\n")
    for vid, status in (("done1", "completed"), ("open1", "pending")):
        bdir = Path(M.bundle_dir(vid))
        bdir.mkdir(parents=True)
        (bdir / "transcript.txt").write_text("spoken words only\n")
        (bdir / "meta.json").write_text(json.dumps({"video_id": vid, "description": "arXiv:2609.04148"}))
    state = _state(done1={"title": "Done", "published": "20260901", "status": "completed",
                          "youtube_note": str(note)},
                   open1={"title": "Open", "status": "pending"})
    assert [f["video_id"] for f in M.backfill_papers(state, dry_run=True)] == ["done1"]
    assert calls == [] and not Path(M.papers_state_file()).exists(), "a dry run registers nothing"
    found = M.backfill_papers(state)
    assert [(f["arxiv_id"], f["video_id"]) for f in found] == [("2609.04148", "done1")]
    assert "[[20260901-done]]" in Path(found[0]["note_path"]).read_text()



# ── #3983: the keyframe leg — sampling, the frame stage, and its failure ─────
#
# All offline: `--frames=<id>` is the only entry, and nothing in `--fetch`, the
# source job or the digest reaches any of it. What is pinned is the contract with
# ffmpeg (one seeked JPEG pass per timestamp, scaled to the resolution cap), the
# cap as a token budget, and the shape of a failure — a video with no downloadable
# media must leave the bundle exactly as it was and `seen.json` untouched.

def _frame_bundle(vid="vid1", **extra):
    """A published bundle on disk (meta.json + transcript.txt); returns its dir."""
    bdir = Path(M.bundle_dir(vid))
    bdir.mkdir(parents=True, exist_ok=True)
    meta = {"video_id": vid, "title": "A Talk", "published": "20261109",
            "url": f"https://www.youtube.com/watch?v={vid}",
            "bundle_dir": str(bdir),
            "meta_path": str(bdir / "meta.json"),
            "transcript_path": str(bdir / "transcript.txt"),
            "transcript_words": 900, "transcript_lines": 40}
    meta.update(extra)
    (bdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (bdir / "transcript.txt").write_text("hello world\n")
    return bdir


def _run_frames_cli(argv):
    """Run main() with argv, capture stdout. Returns (stdout, exception or None)."""
    buf = io.StringIO()
    saved = sys.argv
    sys.argv = ["youtube_channel_monitor.py", *argv]
    try:
        with redirect_stdout(buf):
            M.main()
        return buf.getvalue(), None
    except SystemExit as exc:          # argparse usage error, or a non-zero exit
        return buf.getvalue(), exc
    finally:
        sys.argv = saved


def test_sample_frame_timestamps_takes_one_frame_per_scene_change(tmp_path):
    """≥ 2 detected cuts → exactly those timestamps, sorted and de-duplicated."""
    seen = []

    def probe(path, threshold, log):
        seen.append((path, threshold))
        return [12.5, 2.0, 12.5, 40.0]

    got = M.sample_frame_timestamps("/tmp/x.mp4", duration=300.0, probe=probe)
    assert got == [2.0, 12.5, 40.0], "one per change, ascending, no duplicate"
    assert seen == [("/tmp/x.mp4", M.SCENE_THRESHOLD)], "probed once, at the threshold"


@pytest.mark.parametrize("cuts", [[], [7.0]], ids=["no-cuts", "one-cut-only"])
def test_sample_frame_timestamps_falls_back_to_a_uniform_30s_grid(cuts):
    """Fewer than 2 cuts is the uniform regime — one cut is not a scene list.

    A single cut tells you where *a* change was, not where the slides are, so the
    grid is the honest answer and the pass still sees a whiteboard at 30 s.
    """
    got = M.sample_frame_timestamps("/tmp/x.mp4", duration=100.0,
                                    probe=lambda p, t, l: list(cuts))
    assert got == [0.0, 30.0, 60.0, 90.0]


def test_sample_frame_timestamps_uniform_grid_stops_at_the_cap():
    """10 hours at 30 s is 1200 frames; the cap is 24, and the cap wins.

    The cap is a token budget, not a preference: 1300 estimated tokens per frame on
    the primary's ViT against a 20-image-per-request ceiling.
    """
    assert M.MAX_FRAMES_PER_VIDEO == 24
    got = M.sample_frame_timestamps("/tmp/x.mp4", duration=36000.0)
    assert len(got) == 24 and got[0] == 0.0 and got[-1] == 690.0


def test_sample_frame_timestamps_caps_the_scene_branch_too():
    """The scene regime is capped by the same rule, keeping the earliest cuts."""
    many = [float(i) for i in range(100)]
    got = M.sample_frame_timestamps("/tmp/x.mp4", duration=1e6, cap=5,
                                    probe=lambda p, t, l: many)
    assert got == [0.0, 1.0, 2.0, 3.0, 4.0] and len(got) == 5


def test_sample_frame_timestamps_without_a_duration_samples_frame_zero_only():
    """Unknown duration bounds nothing, so the pass takes one frame, not zero.

    Zero frames would make the caller report failure on a short clip that is
    simply unprobeable; one frame is a real, if thin, answer.
    """
    assert M.sample_frame_timestamps("/tmp/x.mp4", duration=0.0,
                                     probe=lambda p, t, l: []) == [0.0]


def test_probe_scene_changes_escapes_the_filter_comma_and_parses_stamps(monkeypatch):
    """The lavfi graph must survive argv, and ffprobe's framing lines must not.

    `gt(scene\\,0.3)` needs the backslash because a bare comma would separate
    filters instead of arguments; an unescaped graph fails to parse and every video
    silently degrades to the uniform grid.
    """
    calls = []

    class R:
        returncode, stdout = 0, "frame=\n12.5\nframe=\n40\njunk\n"

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return R()

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    assert M.probe_scene_changes("/tmp/a b.mp4") == [12.5, 40.0]
    cmd = calls[0]
    assert cmd[-1] == "movie=/tmp/a b.mp4,select=gt(scene\\,0.3)", \
        "the lavfi graph is ffprobe's LAST argument and its ONLY input"
    assert "/tmp/a b.mp4" not in cmd, (
        "the graph IS the filename under -f lavfi, so naming the file again is a "
        "SECOND input: ffmpeg n9.0.2 exits 1 with \"provided as input filename, but "
        "'movie=…' was already specified\", and rc != 0 is indistinguishable from a "
        "genuinely scene-free video — the scene branch would vanish on every video "
        "with nothing printed but a uniform grid")


def test_probe_scene_changes_returns_empty_list_when_ffprobe_fails(monkeypatch):
    """A missing ffprobe degrades to uniform sampling; it must not raise."""
    def boom(cmd, **kw):
        raise OSError("ffprobe: command not found")

    monkeypatch.setattr(M.subprocess, "run", boom)
    assert M.probe_scene_changes("/tmp/x.mp4") == []
    assert M.probe_duration("/tmp/x.mp4") == 0.0


def test_extract_frames_runs_one_seeked_scaled_jpeg_pass_per_timestamp(tmp_path,
                                                                      monkeypatch):
    """The ffmpeg boundary: `-ss <ts> -i <media> -frames:v 1`, scaled to the cap.

    Irregular scene-change timestamps rule out a single `fps=` filter pass, and a
    frame that fails to extract drops out of the rows rather than failing the deck.
    """
    calls, out_dir = [], tmp_path / "frames"

    class R:
        returncode, stderr = 0, ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 2:
            return R()                      # the middle timestamp fails to seek
        target = Path(cmd[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\xff\xd8\xff")
        return R()

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    rows = M.extract_frames("/tmp/x.mp4", [0.0, 30.0, 60.0], str(out_dir))
    assert [r["path"] for r in rows] == ["frames/000.jpg", "frames/002.jpg"], \
        "the failed frame is skipped, not zero-filled"
    assert [r["ts"] for r in rows] == [0.0, 60.0]
    first = calls[0]
    assert first[first.index("-ss") + 1] == "0.0"
    assert first[first.index("-frames:v") + 1] == "1"
    scale = [a for a in first if a.startswith("scale=")][0]
    assert f"{M.FRAME_RESOLUTION_CAP}:{M.FRAME_RESOLUTION_CAP}" in scale, \
        "sampled at the recorded resolution cap, downscales only"
    assert "-y" in first and not any("base64" in a for a in first)


def test_frame_stage_writes_frames_manifest_and_meta_block(tmp_path, monkeypatch):
    """`--frames=<id>` success path: frames/, frames.jsonl, and the meta.json block.

    The manifest row is a path under the bundle plus a timestamp in seconds and
    nothing else — it is how a later reader knows which frame said what without
    putting a frame in a prompt. The meta block records count, ffmpeg version and
    resolution cap because all three change what "found no literals" will mean.
    """
    bdir = _frame_bundle("vidA")
    media = tmp_path / "media" / "vidA.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"video-bytes")
    monkeypatch.setattr(M, "MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setattr(M, "fetch_media",
                        lambda vid, url, log=print: (str(media), None))
    monkeypatch.setattr(M, "probe_duration", lambda p, log=print: 90.0)
    monkeypatch.setattr(M, "probe_scene_changes",
                        lambda p, threshold=0.3, log=print: [10.0, 40.0])
    monkeypatch.setattr(M, "ffmpeg_version", lambda log=print: "7.1.1")

    def fake_extract(media_path, stamps, frames_dir, **kw):
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        rows = []
        for i, ts in enumerate(stamps):
            Path(frames_dir, f"{i:03d}.jpg").write_bytes(b"\xff\xd8\xff")
            rows.append({"path": f"frames/{i:03d}.jpg", "ts": ts})
        return rows

    monkeypatch.setattr(M, "extract_frames", fake_extract)
    summary, reason = M.run_frame_stage("vidA", log=lambda s: None)
    assert reason is None and summary["frames"] == 2
    assert summary["sampling"] == "scene-changes", "2 cuts is the scene regime"

    rows = [json.loads(l) for l in
            (bdir / "frames.jsonl").read_text().splitlines() if l.strip()]
    assert rows == [{"path": "frames/000.jpg", "ts": 10.0},
                    {"path": "frames/001.jpg", "ts": 40.0}]
    assert sorted(p.name for p in (bdir / "frames").iterdir()) == \
        ["000.jpg", "001.jpg"]

    meta = json.loads((bdir / "meta.json").read_text())
    block = meta["frames"]
    assert block["count"] == 2
    assert block["ffmpeg_version"] == "7.1.1"
    assert block["resolution_cap_px"] == M.FRAME_RESOLUTION_CAP == 1280
    assert block["cap_per_video"] == 24
    assert block["manifest"] == "frames.jsonl" and block["directory"] == "frames/"
    assert meta["transcript_path"] == str(bdir / "transcript.txt"), \
        "a successful frame pass still leaves the transcript fields alone"
    assert not media.exists(), "MEDIA_KEEP is False: the video is not the artifact"


def test_frame_stage_writes_the_uniform_sampling_mode_in_meta(tmp_path, monkeypatch):
    """One cut recorded as `uniform-30s`, so a reader knows the grid, not a slide."""
    bdir = _frame_bundle("vidB")
    media = tmp_path / "media" / "vidB.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"video-bytes")
    monkeypatch.setattr(M, "MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setattr(M, "fetch_media",
                        lambda vid, url, log=print: (str(media), None))
    monkeypatch.setattr(M, "probe_duration", lambda p, log=print: 95.0)
    monkeypatch.setattr(M, "probe_scene_changes",
                        lambda p, threshold=0.3, log=print: [20.0])
    monkeypatch.setattr(M, "ffmpeg_version", lambda log=print: "7.1.1")

    def fake_extract(media_path, stamps, frames_dir, **kw):
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        return [{"path": f"frames/{i:03d}.jpg", "ts": ts}
                for i, ts in enumerate(stamps)]

    monkeypatch.setattr(M, "extract_frames", fake_extract)
    summary, reason = M.run_frame_stage("vidB", log=lambda s: None)
    assert reason is None and summary["sampling"] == "uniform-30s"
    rows = [json.loads(l) for l in
            (bdir / "frames.jsonl").read_text().splitlines() if l.strip()]
    assert [r["ts"] for r in rows] == [0.0, 30.0, 60.0, 90.0]


def test_frame_stage_on_unfetchable_media_writes_nothing(tmp_path, monkeypatch):
    """Media failure: no frames/, no frames.jsonl, meta.json byte-identical.

    This is the clause that keeps the arm honest — a bundle whose video cannot be
    downloaded must be indistinguishable from a bundle the arm never touched, or a
    digest run would see a frames block and reason about an absent screen.
    """
    bdir = _frame_bundle("vidC")
    before = (bdir / "meta.json").read_bytes()

    def failing_fetch(vid, url, log=print):
        return "", "yt-dlp exited 1: video unavailable"

    monkeypatch.setattr(M, "fetch_media", failing_fetch)
    monkeypatch.setattr(M, "probe_duration",
                        lambda p, log=print: pytest.fail("probed after a failed fetch"))
    summary, reason = M.run_frame_stage("vidC", log=lambda s: None)
    assert summary is None
    assert "media unavailable" in reason and "video unavailable" in reason
    assert not (bdir / "frames").exists(), "no frames/ directory"
    assert not (bdir / "frames.jsonl").exists(), "no frame manifest"
    assert (bdir / "meta.json").read_bytes() == before, \
        "meta.json untouched: transcript fields and everything else"


def test_frames_cli_exits_zero_and_leaves_seen_json_untouched(tmp_path, monkeypatch):
    """`--frames=<id>` on unfetchable media: exit 0, the reason in the output, state dead.

    A video with no downloadable media is a fact about that video, not a failed
    command — and this pass has no status to advance, so `seen.json` must be
    byte-identical afterwards. Were it written, a diagnostic run would move a
    video's digest state, which is the difference between a leg and a second
    ingest path.
    """
    # Configure the channel main() will configure FIRST, so the bundle and the
    # seen.json seeded here are the ones the pass actually reads — not sibling
    # files under another channel's sandbox that were never in play.
    M.configure(M.DEFAULT_CHANNEL)
    bdir = _frame_bundle("vidD")
    M.save_state(_state(vidD={"status": "pending", "title": "A Talk"}))
    seen_path = Path(M.STATE_FILE)
    seen_before = seen_path.read_bytes()
    monkeypatch.setattr(M, "fetch_media",
                        lambda v, url, log=print: ("", "yt-dlp is not on PATH"))
    out, exc = _run_frames_cli(["--frames=vidD", "--json"])
    assert exc is None, f"the keyframe pass exited nonzero: {exc} / {out}"
    assert "vidD" in out and "media unavailable" in out
    assert seen_path.read_bytes() == seen_before, "seen.json untouched"
    assert not (bdir / "frames").exists()
    json_lines = [l for l in out.splitlines() if l.startswith(M.JSON_MARK)]
    assert len(json_lines) == 1, f"--json must print exactly one JSON line: {out}"
    payload = json.loads(json_lines[0][len(M.JSON_MARK):])
    assert payload["ok"] is True and payload["seen_json_unchanged"] is True
    assert payload["result"]["frame_reason"].startswith("media unavailable")
    assert payload["result"]["onscreen"] is None, \
        "no on-screen leg runs when no frames were sampled"


def test_build_bundle_itself_never_samples_frames(monkeypatch, tmp_path):
    """The REAL `build_bundle` runs to completion and the leg still does not run.

    The reviewer's point on the first gate: pinning `main()` while stubbing
    `build_bundle` leaves the natural insertion point unpinned — a
    `run_frame_stage(...)` added inside the bundle build would have passed. So
    nothing here stubs `build_bundle`: only its external I/O is replaced (yt-dlp
    metadata, the transcript, GitHub clones, paper registration, the metrics
    snapshot), the bundle is really written, and the assertion is about that
    bundle on disk — no `frames/`, no `frames.jsonl`, no `frames` block, because
    the transcript-only bundle is the control arm of the comparison that decides
    whether this leg survives at all.
    """
    calls = []
    monkeypatch.setattr(M, "run_frame_stage",
                        lambda *a, **k: calls.append(("run_frame_stage",)) or (None, "x"))
    monkeypatch.setattr(M, "run_frames",
                        lambda *a, **k: calls.append(("run_frames",)) or ({}, None))
    monkeypatch.setattr(M, "fetch_media",
                        lambda *a, **k: calls.append(("fetch_media",)) or ("", "no"))
    monkeypatch.setattr(M, "fetch_video_metadata",
                        lambda vid: {"title": "A Talk", "upload_date": "20261109",
                                     "description": "desc"})
    monkeypatch.setattr(M, "fetch_transcript",
                        lambda vid: ("some transcript long enough to bundle " * 20, None))
    # Enrichment returns empty rather than failing: what this pins is that the
    # frame leg is not part of a bundle build, and a stub that raises on the way
    # would abort the build before the assertion could be made.
    monkeypatch.setattr(M, "clone_and_note", lambda url: None)
    monkeypatch.setattr(M, "enrich_papers", lambda ids, video=None: [])
    monkeypatch.setattr(M, "fetch_generic_paper", lambda url: None)
    def capture(bdir):
        # Stands in for the metrics snapshot — which the real function ALWAYS
        # writes, failures included — so the file list below is a claim about the
        # leg's absence, not about a stubbed side effect.
        path = Path(bdir) / "measurements.json"
        path.write_text('{"captured_at": "2026-11-09T00:00:00+00:00", "errors": []}')
        return str(path), {}

    monkeypatch.setattr(M, "capture_measurements", capture)

    meta, err = M.build_bundle("vidG", "A Talk", "20261109", {})
    assert err is None and meta, f"the real bundle build failed: {err}"
    bdir = Path(meta["bundle_dir"])
    assert calls == [], f"the keyframe leg ran from inside build_bundle: {calls}"
    assert not (bdir / "frames").exists() and not (bdir / "frames.jsonl").exists()
    assert "frames" not in json.loads((bdir / "meta.json").read_text()), \
        "a --fetch bundle carries no frames block"
    assert sorted(p.name for p in bdir.iterdir()) == \
        ["measurements.json", "meta.json", "transcript.txt"], \
        "the control arm is still exactly three files"


def test_fetch_wiring_never_touches_the_keyframe_leg(monkeypatch):
    """`--fetch` must stay the transcript-only control arm of #3983's comparison.

    Pinned structurally rather than by reading the code: the frame stage raises if
    it is ever reached from a bundle build, so wiring `--frames` into `--fetch` —
    which would contaminate the measurement that decides whether this leg survives
    — fails here rather than in someone's eval.
    """
    calls = []
    monkeypatch.setattr(M, "run_frame_stage",
                        lambda *a, **k: calls.append(a) or (None, "must not run"))
    monkeypatch.setattr(M, "run_frames",
                        lambda *a, **k: calls.append(a) or (None, "must not run"))
    monkeypatch.setattr(M, "fetch_media",
                        lambda *a, **k: calls.append("media") or ("", "no"))
    monkeypatch.setattr(M, "fetch_transcript",
                        lambda vid: ("some transcript long enough to bundle " * 20, None))
    monkeypatch.setattr(M, "build_bundle",
                        lambda vid, title="", published="", entry=None: (
                            {"video_id": vid, "title": title,
                             "published": published or "20261109",
                             "bundle_dir": "/tmp", "meta_path": "/tmp/meta.json",
                             "transcript_path": "/tmp/transcript.txt"}, None))
    out, exc = _run_frames_cli(["--fetch=vidF"])
    assert exc is None
    assert calls == [], f"the keyframe leg ran from --fetch: {calls}"
