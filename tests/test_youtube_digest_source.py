"""youtube-digest — one video per visible session, and what makes it trustworthy.

The digest and the Lloyd eval moved out of a script that POSTed to the model
into a real session so the transcript is reviewable. Pinned here: the prompt
carries the paths and Alan's adoption rules; disk decides whether a note was
written; a `FILED:` claim is verified; a failed turn is reported to the
script's state (which owns the retry) and a drain is not; the toolbox denies
the shell and the selfmod loop; and the producer interleaves channels under a
bounded queue depth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workers.queue import QueueItem, WorkQueue
from workers.sources import youtube_digest as Y
from workers.sources._common import DrainActive, TurnTimeout, WORKER_SELFMOD_BAN


def _item(payload: dict, item_id: int = 1) -> QueueItem:
    return QueueItem(id=item_id, source=Y.NAME, kind="video", priority=60, payload=payload,
                     dedup_key=None, state="running", attempts=1, enqueued_at="",
                     claimed_at=None, claimed_by=None, completed_at=None, error=None)


def _meta(tmp_path: Path, *, existing: str | None = None) -> dict:
    bdir = tmp_path / "bundles" / "abc123"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "transcript.txt").write_text("words\n" * 50)
    target = existing or str(tmp_path / "vault" / "20260904-second-harness.md")
    return {
        "channel_key": "discover-ai", "channel_handle": "code4AI", "channel_name": "Discover AI",
        "video_id": "abc123", "url": "https://www.youtube.com/watch?v=abc123",
        "title": "Claude Code Improves Massively w/ 2nd Harness", "published": "20260904",
        "bundle_dir": str(bdir), "meta_path": str(bdir / "meta.json"),
        "transcript_path": str(bdir / "transcript.txt"), "transcript_words": 6000,
        "transcript_lines": 420, "entities": {}, "existing_note": existing, "target_note": target,
        "enrichment": {"github": [{"note_path": "/v/github/acme-harness.md"}], "papers": []},
    }


class _Script:
    """Stand-in for the monitor script: records calls, answers by mode."""

    def __init__(self, fetch: dict, pending: dict[str, list] | None = None):
        self.fetch = fetch
        self.pending = pending or {}
        self.calls: list[tuple] = []

    async def __call__(self, channel, *args, timeout):
        self.calls.append((channel, *args))
        mode = args[0]
        if mode == "--fetch":
            return self.fetch
        if mode == "--register-new":
            return {"ok": True, "pending": self.pending.get(channel, [])}
        return {"ok": True}

    def modes(self):
        return [c[1] for c in self.calls]


def _turn(text: str, *, writes: Path | None = None, video_id: str = "abc123",
          raises: Exception | None = None, stop_reason: str = "stop"):
    async def run(prompt, **kwargs):
        run.prompt = prompt
        run.kwargs = kwargs
        if raises is not None:
            raise raises
        if writes is not None:
            writes.parent.mkdir(parents=True, exist_ok=True)
            writes.write_text(f"---\nsegment: knowledge\nvideo_id: {video_id}\n---\n# Note\n\n" + "body " * 200)
        return {"text": text, "session_id": "20260908_youtubed_ab12", "stop_reason": stop_reason,
                "num_turns": 9, "errors": []}
    run.prompt = ""
    run.kwargs = {}
    return run


BLOCK = """\
Done. Here is the summary.

RESULT: written
NOTE: {note}
RELEVANCE: 82
VERDICT: actionable
AREAS: harness, selfmod
SOURCE_KIND: open-source
APPROACH: adopt
IDEA: Run a second, cheaper harness that reviews the primary's tool plan before dispatch
DUPLICATE_OF: none
FILED: #523
"""


@pytest.fixture
def backlog(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "obsidian" / "backlog"
    d.mkdir(parents=True)
    monkeypatch.setattr(Y, "BACKLOG_DIR", d)
    monkeypatch.setattr(Y, "_vault_dirty_paths", lambda: set())
    return d


def _filed(backlog: Path, item_id: int, video_id: str = "abc123", tag: str = "youtube-eval") -> Path:
    p = backlog / f"{item_id}-second-harness.md"
    p.write_text(f"---\nstatus: draft\nboard: lloyd\ntags:\n- {tag}\n- discover-ai\n---\n\n"
                 f"# Evaluate a second harness\n\nSource: https://www.youtube.com/watch?v={video_id}\n")
    return p


# ---------------------------------------------------------------------------
# The RESULT block
# ---------------------------------------------------------------------------


def test_a_well_formed_block_parses():
    p = Y.parse_result(BLOCK.format(note="/v/n.md"))
    assert p["result"] == "written" and p["note"] == "/v/n.md"
    assert p["relevance"] == 82 and p["verdict"] == "actionable"
    assert p["areas"] == ["harness", "selfmod"]
    assert p["source_kind"] == "open-source" and p["approach"] == "adopt"
    assert p["idea"].startswith("Run a second")
    assert p["duplicate_of"] is None and p["filed"] == 523


def test_the_last_block_wins_and_vocabulary_is_enforced():
    text = BLOCK.format(note="/v/a.md") + "\nOn reflection:\n\nRESULT: kept\nNOTE: /v/b.md\nRELEVANCE: 130\nVERDICT: Worth a look\nAREAS: harness, cooking\nSOURCE_KIND: proprietary\nAPPROACH: none\nIDEA: none\nDUPLICATE_OF: #12\nFILED: none\n"
    p = Y.parse_result(text)
    assert p["result"] == "kept" and p["note"] == "/v/b.md"
    assert p["relevance"] == 100, "clamped"
    assert p["verdict"] == "worth_a_look"
    assert p["areas"] == ["harness"], "unknown areas dropped, not failed"
    assert p["source_kind"] is None and p["approach"] == "none"
    assert p["idea"] == "" and p["duplicate_of"] == 12 and p["filed"] is None
    assert Y.parse_result("no block here") is None


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_prompt_carries_the_paths_the_rules_and_the_tracked_items(tmp_path):
    meta = _meta(tmp_path)
    prompt = Y.build_prompt(meta, [{"id": 500, "title": "Adopt WIKISKILL-style skill compilation"}])
    assert meta["transcript_path"] in prompt and meta["meta_path"] in prompt
    assert meta["target_note"] in prompt and str(Y.PROFILE_PATH) in prompt
    assert "channel: code4AI" in prompt and "video_id: abc123" in prompt
    assert "#500 Adopt WIKISKILL" in prompt
    assert "[[acme-harness]]" in prompt
    assert "No note exists" in prompt
    # Alan's rule: open source may be adopted, commercial is recreated.
    assert "open-source" in prompt and "direct adoption" in prompt
    assert "commercial" in prompt and "never adopted" in prompt and "recreating locally" in prompt
    assert "backlog_write_task" in prompt and "youtube-eval" in prompt and "discover-ai" in prompt
    # The first session rated a talk on the assumption that the prefix stayed
    # cached while the live counter read 68.7%: measured claims get checked.
    assert "check the live number first" in prompt and "Live measurements" in prompt
    assert prompt.rstrip().endswith("say why in one line before the block.")
    assert "RESULT: <written|kept|failed>" in prompt


def test_prompt_tells_the_session_about_an_existing_note(tmp_path):
    existing = str(tmp_path / "vault" / "20260904-old.md")
    prompt = Y.build_prompt(_meta(tmp_path, existing=existing), [])
    assert "already exists at that path" in prompt and "RESULT: kept" in prompt
    assert prompt.count(existing) >= 2, "existing_note field and the target path"
    assert "- none yet" in prompt


def test_the_toolbox_denies_the_shell_and_the_loop_but_keeps_the_job():
    for tool in WORKER_SELFMOD_BAN:
        assert tool in Y.DISALLOWED
    for tool in ("Bash", "Edit", "Task", "autonomy_write_task", "research_propose", "http_request"):
        assert tool in Y.DISALLOWED
    for tool in ("Read", "Write", "backlog_write_task", "backlog_tasks", "vault_recall", "http_fetch"):
        assert tool not in Y.DISALLOWED, f"{tool} is the job"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def test_a_good_turn_is_recorded_with_a_verified_filing(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])
    script = _Script({"ok": True, "meta": meta})
    turn = _turn(BLOCK.format(note=note), writes=note)
    _filed(backlog, 523)
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", turn)

    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123", "max_turns": 33}))

    assert result["status"] == "success", result
    assert result["artifact_path"] == str(note)
    assert "actionable (82)" in result["summary"] and "filed #523" in result["summary"]
    assert script.modes() == ["--fetch", "--complete", "--eval-report"]
    complete = script.calls[1]
    assert complete[2] == "abc123" and complete[4] == str(note)
    ev = json.loads(complete[6])
    assert ev["filed"] == 523 and ev["verdict"] == "actionable" and ev["session_id"] == "20260908_youtubed_ab12"
    assert turn.kwargs["inner_voice"] is True, "the whole point: Inner Voice watches it"
    assert turn.kwargs["max_turns"] == 33 and turn.kwargs["source"] == Y.NAME
    assert set(Y.DISALLOWED) <= set(turn.kwargs["extra_disallowed"])
    assert turn.kwargs["title"].startswith("Discover AI: Claude Code")


async def test_an_unverifiable_filing_claim_is_not_recorded_as_filed(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn(BLOCK.format(note=note), writes=note))
    _filed(backlog, 523, video_id="someotherv1d")   # exists, but not about this video

    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))

    assert result["status"] == "success"
    ev = json.loads(script.calls[1][6])
    assert "filed" not in ev and ev["filed_unverified"] == 523
    assert "unverified" in result["summary"]


async def test_no_note_on_disk_is_a_failure_reported_to_the_script(tmp_path, backlog, monkeypatch):
    """The model's text says written; the path is empty. Disk decides, and
    the script's state — which owns the retry — hears about it."""
    meta = _meta(tmp_path)
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session",
                        _turn(BLOCK.format(note=meta["target_note"]), stop_reason="max_turns"))

    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))

    assert result["status"] == "failed"
    assert script.modes() == ["--fetch", "--fail"]
    assert "max_turns" in script.calls[1][4] and "without a note" in script.calls[1][4]
    assert result["meta"]["empty_response"] is False and result["meta"]["stop_reason"] == "max_turns"


async def test_a_note_for_a_different_video_does_not_count(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session",
                        _turn(BLOCK.format(note=note), writes=note, video_id="zzz999"))
    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))
    assert result["status"] == "failed" and "--fail" in script.modes()


async def test_a_fetch_failure_is_reported_without_a_session(tmp_path, backlog, monkeypatch):
    script = _Script({"ok": False, "video_id": "abc123", "error": "Could not fetch transcript: none", "failure_count": 2})
    ran = []

    async def never(*a, **k):
        ran.append(1)

    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", never)
    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))
    assert result["status"] == "failed" and "transcript" in result["summary"]
    assert ran == [] and script.modes() == ["--fetch"], "the script already counted the attempt"
    assert result["meta"]["failure_count"] == 2


async def test_a_drain_is_skipped_and_not_counted_against_the_video(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", raises=DrainActive("landing")))
    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))
    assert result["status"] == "skipped"
    assert script.modes() == ["--fetch"], "no --fail: the entry stays fetched and is re-offered"


async def test_a_turn_timeout_is_a_counted_failure(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", raises=TurnTimeout("1800s")))
    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))
    assert result["status"] == "failed" and result["meta"]["turn_timeout"] is True
    assert script.modes() == ["--fetch", "--fail"] and "timeout" in script.calls[1][4]


async def test_a_kept_note_is_a_success(tmp_path, backlog, monkeypatch):
    existing = tmp_path / "vault" / "20260904-old.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("---\nsegment: knowledge\nvideo_id: abc123\n---\n# Old\n\n" + "body " * 200)
    meta = _meta(tmp_path, existing=str(existing))
    script = _Script({"ok": True, "meta": meta})
    text = BLOCK.format(note=existing).replace("RESULT: written", "RESULT: kept").replace("FILED: #523", "FILED: none")
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn(text))
    result = await Y.execute(_item({"channel": "ai-engineer", "video_id": "abc123"}))
    assert result["status"] == "success" and "note kept" in result["summary"]
    assert json.loads(script.calls[1][6])["note_result"] == "kept"


# ---------------------------------------------------------------------------
# Tracked items and the producer
# ---------------------------------------------------------------------------


def test_tracked_items_reads_only_this_evals_items(backlog):
    _filed(backlog, 510)
    _filed(backlog, 511, tag="spawned-by-triage")
    (backlog / "512-unrelated.md").write_text("---\ntags: [youtube-eval, harness]\n---\n\n# Flow style tags\n")
    got = Y.tracked_items()
    assert [t["id"] for t in got] == [510, 512]
    assert got[0]["title"] == "Evaluate a second harness"
    assert Y._filed_item_exists(510, "abc123") and not Y._filed_item_exists(510, "other")
    assert not Y._filed_item_exists(999, "abc123")


async def test_the_producer_interleaves_channels_under_a_bounded_depth(tmp_path, monkeypatch):
    queue = WorkQueue(tmp_path / "workers.db")
    pending = {
        "ai-engineer": [{"video_id": f"ae{i}", "title": f"AE {i}", "published": "20260908"} for i in range(20)],
        "discover-ai": [{"video_id": f"da{i}", "title": f"DA {i}", "published": "20260908"} for i in range(5)],
    }
    script = _Script({}, pending=pending)
    monkeypatch.setattr(Y, "_script", script)
    cfg = {"batch": 4, "channels": ["ai-engineer", "discover-ai"], "max_turns": 40}

    await Y.enqueue_if_due(queue, cfg)
    items = queue.list_items(source=Y.NAME)
    ids = [i.payload["video_id"] for i in items]
    assert sorted(ids) == ["ae0", "ae1", "da0", "da1"], "two each, not four AI Engineer"
    assert all(i.dedup_key == f"youtube-digest:{i.payload['channel']}:{i.payload['video_id']}" for i in items)
    assert all(i.payload["max_turns"] == 40 for i in items)

    # Full: the next tick adds nothing and does not even list the channels.
    calls_before = len(script.calls)
    await Y.enqueue_if_due(queue, cfg)
    assert len(queue.list_items(source=Y.NAME)) == 4
    assert len(script.calls) == calls_before


def test_execute_does_not_block_the_event_loop():
    import inspect
    src = inspect.getsource(Y.execute) + inspect.getsource(Y.enqueue_if_due)
    for pattern in ("subprocess.run", "subprocess.check_output", "time.sleep(", "urlopen("):
        assert pattern not in src
