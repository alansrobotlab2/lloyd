"""youtube-digest — one video per visible session, and what makes it trustworthy.

The digest and the Lloyd eval moved out of a script that POSTed to the model
into a real session so the transcript is reviewable. Pinned here: the prompt
carries the paths and Alan's adoption rules; disk decides whether a note was
written; a `FILED:` claim is verified; a failed turn is reported to the
script's state (which owns the retry) and a drain is not; the toolbox denies
the shell and the automod loop; and the producer interleaves channels under a
bounded queue depth.

Since #737 the protocol itself is not in this file: `build_prompt` loads the
vault's `youtube-digest` skill and renders it with the per-bundle `<video>`
task block, so the last section pins the loader, the one brace-free placeholder
surface, and the failure when the skill is gone.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from workers.queue import QueueItem, WorkQueue
from workers.sources import youtube_digest as Y
from workers.sources._common import DrainActive, TurnTimeout, WORKER_AUTOMOD_BAN


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
        "measurements_path": str(bdir / "measurements.json"),
        "measurements_summary": "prefix-cache hit rate since boot 68.7%; KV cache 79% used",
    }


class _Script:
    """Stand-in for the monitor script: records calls, answers by mode."""

    def __init__(self, fetch: dict, pending: dict[str, list] | None = None):
        self.fetch = fetch
        self.pending = pending or {}
        self.calls: list[tuple] = []

    async def __call__(self, channel, *args, timeout):
        self.calls.append((channel, *args))
        mode = args[0].split("=", 1)[0]
        if mode == "--fetch":
            return self.fetch
        if mode == "--register-new":
            return {"ok": True, "pending": self.pending.get(channel, [])}
        return {"ok": True}

    def modes(self):
        return [c[1].split("=", 1)[0] for c in self.calls]

    def opt(self, call_index, name):
        """The value of `--name` on one recorded call, which rides attached
        to its option — see `_script`. Returns None when it is absent."""
        for arg in self.calls[call_index][1:]:
            if arg.startswith(f"{name}="):
                return arg[len(name) + 1:]
        return None


def _turn(text: str, *, writes: Path | None = None, video_id: str = "abc123",
          raises: Exception | None = None, stop_reason: str = "stop",
          structured: dict | None = None, structured_error: str = ""):
    async def run(prompt, **kwargs):
        run.prompt = prompt
        run.kwargs = kwargs
        if raises is not None:
            raise raises
        if writes is not None:
            writes.parent.mkdir(parents=True, exist_ok=True)
            writes.write_text(f"---\nsegment: knowledge\nvideo_id: {video_id}\n---\n# Note\n\n" + "body " * 200)
        return {"text": text, "session_id": "20260908_youtubed_ab12", "stop_reason": stop_reason,
                "num_turns": 9, "errors": [], "structured": structured,
                "structured_error": structured_error}
    run.prompt = ""
    run.kwargs = {}
    return run


BLOCK = """\
Done. Here is the summary.

RESULT: written
NOTE: {note}
RELEVANCE: 82
VERDICT: actionable
AREAS: harness, automod
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
    assert p["areas"] == ["harness", "automod"]
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
    assert "check the live number first" in prompt
    assert meta["measurements_path"] in prompt and "68.7%" in prompt
    assert "cannot reach loopback" in prompt
    assert prompt.rstrip().endswith("say why in one line before the block.")
    assert "RESULT: <written|kept|failed>" in prompt


def test_prompt_tells_the_session_about_an_existing_note(tmp_path):
    existing = str(tmp_path / "vault" / "20260904-old.md")
    prompt = Y.build_prompt(_meta(tmp_path, existing=existing), [])
    assert "already exists at that path" in prompt and "RESULT: kept" in prompt
    assert prompt.count(existing) >= 2, "existing_note field and the target path"
    assert "- none yet" in prompt


def test_the_toolbox_denies_the_shell_and_the_loop_but_keeps_the_job():
    for tool in WORKER_AUTOMOD_BAN:
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
    assert script.opt(1, "--complete") == "abc123" and script.opt(1, "--note") == str(note)
    ev = json.loads(script.opt(1, "--eval-json"))
    assert ev["filed"] == 523 and ev["verdict"] == "actionable" and ev["session_id"] == "20260908_youtubed_ab12"
    # Resolved from `workers.sources.youtube-digest.inner_voice`, not passed
    # by the source — one reader for the switch, or the config is decoration.
    assert "inner_voice" not in turn.kwargs
    from workers.sources._common import source_inner_voice
    # Off since cut 1 of senses-not-supervision: the turn is still a real,
    # recorded session — that was the point, and it holds — but the observer
    # no longer watches unattended turns. The source passes no override, so
    # config is the whole of it.
    assert source_inner_voice(Y.NAME) is False
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
    ev = json.loads(script.opt(1, "--eval-json"))
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
    reason = script.opt(1, "--reason")
    assert "max_turns" in reason and "without a note" in reason
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


async def test_a_fetch_crash_is_counted_so_the_row_cannot_spin_forever(tmp_path, backlog, monkeypatch):
    """An `ok: False` was already counted by the script. A *crash* was not —
    the script exited before it marked anything — so the row stays `pending`
    and `pending_entries` re-offers it every tick with nothing about the next
    attempt different. Counting it puts the row behind the retry bound."""
    calls = []

    async def script(channel, *args, timeout):
        calls.append((channel, *args))
        if args[0].startswith("--fetch"):
            raise Y.ScriptError("--fetch rc=2: expected one argument")
        return {"ok": True}

    ran = []

    async def never(*a, **k):
        ran.append(1)

    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", never)
    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))

    assert result["status"] == "failed" and result["meta"]["fetch_crashed"] is True
    assert ran == [], "no session for a video whose bundle does not exist"
    assert [c[1].split("=", 1)[0] for c in calls] == ["--fetch", "--fail"]
    assert "fetch crashed" in calls[1][2]


class _Subprocess:
    """Stand-in for the monitor script as a *process*, so the real `_script`
    builds the argv. Records every command line it is handed."""

    def __init__(self, fetch: dict):
        self.fetch = fetch
        self.cmds: list[list[str]] = []

    async def __call__(self, *cmd, stdout=None, stderr=None):
        self.cmds.append(list(cmd))
        payload = self.fetch if any(a.startswith("--fetch") for a in cmd) else {"ok": True}

        class _Proc:
            returncode = 0

            async def communicate(self):
                return (f"{Y.JSON_MARK}{json.dumps(payload)}\n".encode(), b"")

        return _Proc()

    def dashed(self) -> list[str]:
        """Every argument argparse would read as a flag rather than a value."""
        return [a for cmd in self.cmds for a in cmd
                if a.startswith("-") and not a.startswith("--")]

    def modes(self) -> list[str]:
        return [a.split("=", 1)[0] for cmd in self.cmds for a in cmd
                if a.startswith("--") and a not in ("--channel", "--json")
                and not a.startswith(("--note", "--reason", "--eval-json"))]


async def test_a_video_id_starting_with_a_dash_reaches_the_script(tmp_path, backlog, monkeypatch):
    """YouTube ids are base64url, so roughly one in forty begins with `-`. As
    its own argv token argparse reads it as a flag and the script exits 2 with
    its usage text — the video is not failed, it is unattemptable. Seven such
    ids sat at the head of the two channels' newest-first queues on
    2026-09-09 and burned 89 runs; Discover AI stopped draining entirely.

    The real `_script` builds the argv here, so this covers every option it
    hands a value: `--fetch` on the way in and `--fail` on the way out.
    """
    proc = _Subprocess({"ok": True, "meta": _meta(tmp_path)})
    monkeypatch.setattr(Y.asyncio, "create_subprocess_exec", proc)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", raises=TurnTimeout("1800s")))

    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "-f_iTetjgvE"}))

    assert result["status"] == "failed" and result["meta"]["turn_timeout"] is True
    assert proc.modes() == ["--fetch", "--fail"]
    assert any("--fetch=-f_iTetjgvE" in cmd for cmd in proc.cmds), proc.cmds
    assert any("--fail=-f_iTetjgvE" in cmd for cmd in proc.cmds), proc.cmds
    assert proc.dashed() == [], f"argparse reads these as flags: {proc.dashed()}"


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
    assert script.modes() == ["--fetch", "--fail"] and "timeout" in script.opt(1, "--reason")


async def test_an_infra_shaped_turn_is_not_counted_against_the_video(tmp_path, backlog, monkeypatch):
    """No text and no stop reason: the engine was unreachable. The row stays
    fetched (no --fail), and the run is recorded as an infra failure."""
    existing = tmp_path / "vault" / "20260904-old.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("---\nsegment: knowledge\nvideo_id: abc123\n---\n# Old\n\n" + "body " * 200)
    meta = _meta(tmp_path, existing=str(existing))
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", stop_reason=None))
    result = await Y.execute(_item({"channel": "ai-engineer", "video_id": "abc123"}))
    assert result["status"] == "failed" and result["meta"]["infra"] is True
    assert script.modes() == ["--fetch"], "an outage must not burn the video's retries"


async def test_a_pre_existing_note_is_not_proof_the_turn_ran(tmp_path, backlog, monkeypatch):
    """The 2026-09-09 misrecording: the engine was down, the turn returned
    prose with no RESULT block over an old note, and the source called it
    completed because the file was there."""
    existing = tmp_path / "vault" / "20260904-old.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("---\nsegment: knowledge\nvideo_id: abc123\n---\n# Old\n\n" + "body " * 200)
    meta = _meta(tmp_path, existing=str(existing))
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session",
                        _turn("I could not finish.", stop_reason="max_turns"))
    result = await Y.execute(_item({"channel": "ai-engineer", "video_id": "abc123"}))
    assert result["status"] == "failed"
    assert script.modes() == ["--fetch", "--fail"] and "pre-existing note" in script.opt(1, "--reason")


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
    assert json.loads(script.opt(1, "--eval-json"))["note_result"] == "kept"


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


def test_a_filing_merged_into_an_existing_item_still_verifies(tmp_path):
    """`youtube-eval` writes merge like the loop's own since 2026-09-14. The
    item merged into need not carry the eval tag, and the merged text is at
    the end of the file, past the 20 kB head the check read — so a merged
    filing was recorded `filed_unverified`, "no such item on disk"."""
    d = tmp_path
    (d / "10-existing.md").write_text(
        "---\nstatus: draft\ntags: [backlog]\n---\n\n# Existing\n\n" + "x" * 30_000
        + "\n\n## Merged finding — 2026-09-14\n\nFrom video abc123XYZ: the idea.\n")
    assert Y._filed_item_exists(10, "abc123XYZ", backlog_dir=d)
    assert not Y._filed_item_exists(10, "other0video", backlog_dir=d)
    (d / "11-plain.md").write_text("---\nstatus: draft\ntags: [backlog]\n---\n\n# P\n\nabc123XYZ\n")
    assert not Y._filed_item_exists(11, "abc123XYZ", backlog_dir=d), "neither the tag nor a merge"


# ---------------------------------------------------------------------------
# The structured verdict (#710): the finalizer's object first, the block after
# ---------------------------------------------------------------------------


def _obj(result="written", **kw):
    base = {"result": result, "note": "n.md", "relevance": 140, "verdict": "worth_a_look",
            "areas": ["memory", "retrieval"], "source_kind": "paper",
            "approach": "experiment", "idea": "none", "duplicate_of": "none",
            "filed": "none"}
    base.update(kw)
    return base


def _old_note(tmp_path: Path) -> Path:
    existing = tmp_path / "vault" / "20260904-old.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("---\nsegment: knowledge\nvideo_id: abc123\n---\n# Old\n\n" + "body " * 200)
    return existing


def test_the_schema_enums_are_the_parsers_vocabularies():
    props = Y.RESULT_SCHEMA["properties"]
    assert props["result"]["enum"] == ["written", "kept", "failed"] == list(Y._RESULTS)
    assert props["verdict"]["enum"] == list(Y.VERDICTS)
    assert props["source_kind"]["enum"] == list(Y.SOURCE_KINDS)
    assert props["approach"]["enum"] == list(Y.APPROACHES)
    assert props["areas"]["items"]["enum"] == list(Y.AREAS)
    assert Y.RESULT_SCHEMA["additionalProperties"] is False
    assert set(Y.RESULT_SCHEMA["required"]) == set(props)
    # Required, not forbidden (#2444). `_shape` slices nothing, so there was no
    # Python clamp for these caps to interfere with: they exist to turn a degenerate
    # draw into a reported divergence at a token count instead of advice to raise
    # harness.finalizer.max_tokens, and are pinned at their declared values in
    # tests/test_automod_schema_bounds.py.
    from app.harness.finalizer import _schema_is_bounded
    assert _schema_is_bounded(Y.RESULT_SCHEMA), "an open string leaf is back"
    assert props["note"]["maxLength"] == Y.DIGEST_NOTE_MAX == 400, props["note"]
    assert props["idea"]["maxLength"] == Y.DIGEST_IDEA_MAX == 400, props["idea"]


def test_the_structured_object_goes_through_the_same_clamps():
    got = Y.parse_verdict("", _obj(filed="#523", duplicate_of="none"))
    assert got["source"] == "structured"
    assert got["result"] == "written" and got["relevance"] == 100
    assert got["areas"] == ["memory", "retrieval"] and got["filed"] == 523
    assert got["duplicate_of"] is None and got["idea"] == ""


async def test_a_structured_verdict_wins_and_the_regex_is_never_reached(
        tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])
    script = _Script({"ok": True, "meta": meta})
    turn = _turn("prose with no block", writes=note, structured=_obj())
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", turn)

    def boom(text):
        raise AssertionError("parse_result reached with a structured verdict present")
    monkeypatch.setattr(Y, "parse_result", boom)

    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))
    assert result["status"] == "success"
    assert turn.kwargs["final_schema"] is Y.RESULT_SCHEMA
    assert result["meta"]["verdict_source"] == "structured"
    assert json.loads(script.opt(1, "--eval-json"))["verdict"] == "worth_a_look"


async def test_no_object_falls_back_to_the_block_and_says_so(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])
    script = _Script({"ok": True, "meta": meta})
    _filed(backlog, 523)
    seen = []
    real = Y.parse_result
    monkeypatch.setattr(Y, "parse_result", lambda text: seen.append(text) or real(text))
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn(
        BLOCK.format(note=note), writes=note,
        structured_error="finalizer failed: output is not JSON"))

    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))
    assert seen and result["status"] == "success"
    assert result["meta"]["verdict_source"] == "regex"
    assert result["meta"]["structured_error"] == "finalizer failed: output is not JSON"


async def test_the_kill_switch_rides_in_the_payload(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])
    monkeypatch.setattr(Y, "_script", _Script({"ok": True, "meta": meta}))
    turn = _turn(BLOCK.format(note=note), writes=note)
    monkeypatch.setattr(Y, "run_prompt_in_session", turn)
    await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123",
                           "structured_verdict": False}))
    assert turn.kwargs["final_schema"] is None


async def test_an_unreadable_verdict_over_an_old_note_is_not_a_turn_that_ran(
        tmp_path, backlog, monkeypatch):
    """The finalizer ran (so the stop was clean) and neither it nor the text
    produced a known outcome. "A clean stop with text" used to pass exactly
    this turn and record it completed over a note it never wrote."""
    existing = _old_note(tmp_path)
    meta = _meta(tmp_path, existing=str(existing))
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn(
        "RESULT: probably_fine\nVERDICT: background\n", stop_reason="stop",
        structured_error="finalizer failed: output is not JSON"))

    result = await Y.execute(_item({"channel": "ai-engineer", "video_id": "abc123"}))
    assert result["status"] == "failed"
    assert script.modes() == ["--fetch", "--fail"]


async def test_a_clean_stop_without_the_finalizer_keeps_the_old_rule(
        tmp_path, backlog, monkeypatch):
    """Switched off, nothing about the gate moves: a clean stop with text over
    an old note still reads as a turn that ran."""
    existing = _old_note(tmp_path)
    meta = _meta(tmp_path, existing=str(existing))
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("Kept the note as it was."))

    result = await Y.execute(_item({"channel": "ai-engineer", "video_id": "abc123",
                                    "structured_verdict": False}))
    assert result["status"] == "success"


async def test_a_kept_note_from_the_object_is_a_success(tmp_path, backlog, monkeypatch):
    existing = _old_note(tmp_path)
    meta = _meta(tmp_path, existing=str(existing))
    script = _Script({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn(
        "Left it.", structured=_obj("kept", note=str(existing))))

    result = await Y.execute(_item({"channel": "ai-engineer", "video_id": "abc123"}))
    assert result["status"] == "success" and "note kept" in result["summary"]
    assert result["meta"]["verdict_source"] == "structured"


async def test_the_kill_switch_is_read_at_enqueue_time(tmp_path, monkeypatch):
    queue = WorkQueue(tmp_path / "workers.db")

    async def script(channel, *args, timeout):
        return {"ok": True, "pending": [{"video_id": "v1", "title": "t"}]}
    monkeypatch.setattr(Y, "_script", script)
    await Y.enqueue_if_due(queue, {"channels": ["discover-ai"], "structured_verdict": False})
    items = queue.list_items(source=Y.NAME)
    assert items and items[0].payload["structured_verdict"] is False


# ---------------------------------------------------------------------------
# The protocol is the vault skill, not this file (#737)
# ---------------------------------------------------------------------------

SKILL_DIR = Path.home() / "obsidian" / "skills" / Y.SKILL

#: The per-bundle values the task block interpolates. These are the only
#: placeholders allowed anywhere in the digest prompt: the protocol itself is
#: loaded unformatted from the vault skill, so a token added on the vault side
#: would reach the model as a literal `{token}` and no `.format()` would ever
#: fill it in.
PLACEHOLDERS = {
    "channel_name", "channel_handle", "channel_key", "title", "video_id", "url",
    "published", "transcript_path", "transcript_words", "transcript_lines",
    "meta_path", "existing_note", "enrichment", "target_note", "note_stem",
    "existing_note_instruction", "measurements_path", "measurements_summary",
    "profile_path", "tracked", "eval_tag", "areas",
}


def test_the_digest_protocol_is_a_vault_skill_the_real_loader_accepts():
    """The skill exists, loads, and holds the rules that decide a run.

    `skill_load_defect` is the same check `automod_vault_land` runs over a
    touched `skills/**` path before it commits, so passing it here is the same
    fact the landing route asserts — front matter parses, `status` is not a
    quarantine, the file reads. The three rules below are the ones that were
    added to the inline prompt one incident at a time: write the note then read
    it back, verify the tail is not cut off, and the shared prose ban-list.
    """
    from agent_mcp.skills import _parse_frontmatter, skill_load_defect

    assert SKILL_DIR.is_dir(), f"skills/{Y.SKILL} is missing"
    assert skill_load_defect(SKILL_DIR) is None, f"skills/{Y.SKILL} does not load"

    text = (SKILL_DIR / "SKILL.md").read_text()
    front = _parse_frontmatter(text)[0]
    assert front.get("name") == Y.SKILL, f"front matter name is {front.get('name')!r}"
    assert front.get("type") == "skill", f"front matter type is {front.get('type')!r}"
    assert str(front.get("description") or "").strip(), "front matter needs a description"

    lowered = text.lower()
    assert "read the note back" in lowered, "write-then-read-back rule left the protocol"
    assert "verify the tail" in lowered, "tail verification rule left the protocol"
    assert "Prose Rules" in text, "the prose ban-list left the protocol"
    for phrase in ("In this video", "delve", "may potentially"):
        assert phrase in text, f"the ban-list must name `{phrase}` concretely"


def test_the_protocol_reaches_the_prompt_through_the_loader_at_build_time(tmp_path, monkeypatch):
    """`build_prompt` reads the skill by slug every time it renders.

    The old shape baked the protocol into a Python string literal, so no vault
    edit could ever change what a digest turn was told. A sentinel from a
    stubbed loader is what proves the read happens here and now: a leftover
    constant would satisfy every content assertion above and still be dead code.
    """
    seen: list[str] = []

    def fake_load(slug: str):
        seen.append(slug)
        return "PROTOCOL FROM THE VAULT SKILL"

    monkeypatch.setattr("app.autonomy._load_skill_content", fake_load)
    prompt = Y.build_prompt(_meta(tmp_path), [])

    assert seen == [Y.SKILL], f"expected one load of {Y.SKILL!r}, got {seen}"
    assert "PROTOCOL FROM THE VAULT SKILL" in prompt
    assert prompt.startswith(f'[SYSTEM: You are running the "{Y.NAME}" worker job')


def test_the_task_block_supplies_all_22_placeholders_and_renders_them(tmp_path):
    """Every per-bundle value is supplied, and none survives unrendered.

    `build_prompt` used to `.format()` the whole protocol, which is why 467
    runs a week depended on a vault edit never containing a stray brace. Now
    only the task block is formatted, so the brace surface is this file's and
    the failure mode is a `KeyError` here rather than a protocol the model
    cannot read.
    """
    meta = _meta(tmp_path)
    got = set(re.findall(r"\{(\w+)\}", Y.TASK_BLOCK))
    assert got == PLACEHOLDERS, (
        f"task block placeholders differ from the {len(PLACEHOLDERS)} the renderer "
        f"supplies: missing={sorted(PLACEHOLDERS - got)} extra={sorted(got - PLACEHOLDERS)}")

    prompt = Y.build_prompt(meta, [{"id": 500, "title": "Adopt WIKISKILL-style skill compilation"}])
    for name in sorted(PLACEHOLDERS):
        assert "{" + name + "}" not in prompt, f"{{{name}}} reached the prompt unrendered"
    assert "{" not in prompt and "}" not in prompt, (
        "a brace in the rendered prompt means the vault skill grew a placeholder "
        "shaped token; the skill body is not formatted, so nothing would fill it")
    assert meta["target_note"] in prompt and meta["transcript_path"] in prompt
    assert "#500 Adopt WIKISKILL" in prompt and str(Y.PROFILE_PATH) in prompt


def test_the_vault_skill_body_carries_no_placeholder(tmp_path):
    """The other half of the brace rule, checked on the vault side.

    The loader hands the skill text over unformatted, so `{anything}` here is
    never a format field — it is a token the model reads literally. Cheaper to
    fail this test than to fail 467 runs a week.
    """
    body = (SKILL_DIR / "SKILL.md").read_text()
    assert "{" not in body and "}" not in body, (
        f"skills/{Y.SKILL}/SKILL.md carries a brace; per-video values belong in "
        "the task block that workers/sources/youtube_digest.py renders")


@pytest.mark.parametrize("content", [None, "", "   \n"],
                         ids=["absent", "empty", "whitespace"])
async def test_a_missing_skill_fails_the_run_before_the_session(tmp_path, backlog, monkeypatch,
                                                                content):
    """Clause 5, at the seam: no protocol means no dispatch.

    The fallback that was here before made a missing skill invisible — the run
    quietly used a stale copy of the rules and every vault edit to the protocol
    stopped mattering, which is the same silence this item was filed for. A
    named failure is what turns "the skill went missing" into a run record
    instead of a week of digests nobody can explain.
    """
    dispatched: list[str] = []
    monkeypatch.setattr(Y, "_script", _Script({"ok": True, "meta": _meta(tmp_path)}))

    async def fake_session(prompt, **kwargs):
        dispatched.append(prompt)
        return {"text": "", "session_id": "s1", "stop_reason": "stop",
                "num_turns": 1, "errors": []}

    monkeypatch.setattr(Y, "run_prompt_in_session", fake_session)
    monkeypatch.setattr("app.autonomy._load_skill_content", lambda slug: content)

    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))

    assert dispatched == [], "a protocol-less prompt was dispatched to a session"
    assert result["status"] == "failed"
    assert Y.SKILL in result["summary"], f"failure must name the skill: {result['summary']}"
    assert result["meta"].get("skill_missing") is True


# ---------------------------------------------------------------------------
# #1606 — a failure summary is one line
# ---------------------------------------------------------------------------
#
# youtube-digest is the source that pastes a fetched script's stderr into its
# summary verbatim. All 6 of its 500-character rows in `runs` are
# `status='failed'`, and all 6 carry 6 newlines — 7 physical lines in a column
# every reader renders as one record ("ai-engineer g0vqT_wZtXA: Could not fetch transcript:
# Primary: Traceback (most recent call last):\n  File "<string>", line 5, in
# <module>…"), split three each across two videos of the same channel. A
# word-boundary cut at the cap would not have touched those, which is why the
# source now collapses its own whitespace and leaves the length to
# `queue._insert_run`.


async def test_a_traceback_failure_is_recorded_as_one_line_naming_the_video(backlog,
                                                                             tmp_path,
                                                                             monkeypatch):
    """Clause 5 of #1606, through `execute` — not through the helper doing it.

    The fetch script reports the upstream exception as its stderr, verbatim and
    multi-line, and the site was `f"{channel} {video_id}: {why}"[:500]`: a slice
    that flattened nothing, so the traceback's own layout went into the row.
    """
    from workers.pool import normalize_result
    from workers.queue import SUMMARY_MAX_CHARS

    traceback_ = (
        'Could not fetch transcript: Primary: Traceback (most recent call last):\n'
        '  File "<string>", line 5, in <module>\n'
        '    print(get_transcript(sys.argv[1]))\n'
        '  File "/home/alansrobotlab/.venvs/ytdlp/lib/python3.12/site-packages/'
        'youtube_transcript_api/_api.py", line 72, in list\n'
        '    return self.list(language_codes=language_codes)\n'
        'TranscriptsDisabled: \n'
        'could not retrieve a transcript for the video\n'
        'please ask the author to fix this'
    )
    # Six newlines is what the stored rows hold — 7 physical lines at exactly 500
    # characters, the seventh cut off by the cap — and the raw stderr the script
    # returns is longer still, which is why the fixture carries this many.
    assert traceback_.count("\n") == 7, "the fixture must really be multi-line"

    script = _Script({"ok": False, "error": traceback_})
    monkeypatch.setattr(Y, "_script", script)

    item = _item({"channel": "ai-engineer", "video_id": "g0vqT_wZtXA"})
    result = await Y.execute(item)
    assert result["status"] == "failed", \
        f"a failed fetch must fail closed, got {result['status']}"

    # What the source hands the pool, and what the pool stores, both one line.
    assert "\n" not in result["summary"], f"multi-line left the source: {result['summary']!r}"
    stored = normalize_result(_item({"channel": "ai-engineer",
                                        "video_id": "g0vqT_wZtXA"}),
                          result)["summary"]
    assert "\n" not in stored
    assert len(stored) <= SUMMARY_MAX_CHARS
    assert "ai-engineer" in stored and "g0vqT_wZtXA" in stored, \
        "a one-line summary that does not say which video is a worse record: " + stored
    assert "Could not fetch transcript" in stored and "TranscriptsDisabled" in stored, \
        "flattening the layout must keep the reason: " + stored
    assert "  " not in stored, f"traceback indentation survived: {stored!r}"


async def test_a_missing_skill_protocol_failure_is_one_line_too(backlog, tmp_path,
                                                                monkeypatch):
    """The second failure site, reached through the transcript-fetch branch.

    `SkillProtocolMissing` is raised with a message the code itself writes over
    several lines, so this is not an upstream string the source could not predict:
    the site was `f"{channel} {video_id}: {exc}"[:500]` and stored the newlines.
    """
    from workers.pool import normalize_result

    meta = _meta(tmp_path)
    script = _Script({"ok": True, "meta": meta, "transcript": "word " * 900,
                      "subs_lang": "en", "subs_kind": "asr", "duration": "12:34",
                      "title": "What is LLM"})
    monkeypatch.setattr(Y, "_script", script)

    def explode(meta_arg, tracked, **kw):
        raise Y.SkillProtocolMissing(
            "youtube-digest: the worker protocol is missing\n"
            "  the skill is at ~/obsidian/skills/youtube-digest/SKILL.md\n"
            "  and it is empty"
        )

    monkeypatch.setattr(Y, "build_prompt", explode)

    result = await Y.execute(_item({"channel": "ai-engineer", "video_id": "abc123"}))
    assert result["status"] == "failed"
    assert result["meta"].get("skill_missing") is True

    stored = normalize_result(_item({"channel": "ai-engineer", "video_id": "abc123"}),
                              result)["summary"]
    assert "\n" not in stored, f"a source-written newline reached the row: {stored!r}"
    assert "ai-engineer" in stored and "abc123" in stored
    assert "protocol is missing" in stored
    assert "  " not in stored


# ---------------------------------------------------------------------------
# Papers: a second kind of row, read in its own session
# ---------------------------------------------------------------------------
#
# Until 2026-10-09 a cited paper got, at best, an abstract stub, and in fact
# got nothing: the script looked for arXiv ids in the transcript, and across
# 517 bundles there were none. Two things are pinned here. The video turn
# names its papers in the RESULT block (most of Discover AI's descriptions
# give a title and no id, so no regex can), and the source hands them to the
# script after `--complete`. And a `kind="paper"` row runs the same
# fetch → session → disk-decides → report path as a video, files nothing, and
# is a success only when the placeholder has become a full note.

PAPER_SKILL_TEXT = "# Paper Digest\n\nRead the paper, replace the placeholder, end with the RESULT block.\n"
PAPER_HEADER = (
    "---\ntype: notes\narxiv_id: '2609.04148'\ndigest: full\n"
    'title: "Terminal-Universe: Turning {Agent} Trajectories into Environments"\n'
    "tags:\n- paper\n---\n\n# Terminal-Universe — paper note (arXiv 2609.04148)\n")


@pytest.fixture
def paper_skill(monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr("app.autonomy._load_skill_content",
                        lambda slug: asked.append(slug) or PAPER_SKILL_TEXT)
    monkeypatch.setattr(Y, "_vault_dirty_paths", lambda: set())
    return asked


def _paper_meta(tmp_path: Path) -> dict:
    bdir = tmp_path / "papers" / "bundles" / "2609.04148"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "paper.txt").write_text("a line of the paper\n" * 900)
    note = tmp_path / "vault" / "papers" / "2609.04148-terminal-universe.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(PAPER_HEADER.replace("digest: full", "digest: abstract")
                    + "\n## Abstract\n\n" + "abstract " * 300 + "\n\n## Discussed in\n\n- [[20260920-qwen-learns]]\n")
    return {
        "arxiv_id": "2609.04148", "url": "https://arxiv.org/abs/2609.04148",
        "title": "Terminal-Universe: Turning {Agent} Trajectories into Environments",
        "authors": ["Jie Wu", "Zhenru Zhang"], "published": "2026-09-03",
        "bundle_dir": str(bdir), "meta_path": str(bdir / "meta.json"),
        "text_path": str(bdir / "paper.txt"), "text_source": "html", "text_words": 4500,
        "text_lines": 900, "text_truncated": False, "target_note": str(note),
        "note_header": PAPER_HEADER,
        "discussed": ["- [[20260920-qwen-learns]] — Discover AI, 20260920"],
    }


class _PaperScript(_Script):
    def __init__(self, fetch: dict | None = None, *, papers=None, registered=None, **kw):
        super().__init__(fetch or {}, **kw)
        self.paper_fetch = fetch
        self.papers = papers or []
        self.registered = registered

    async def __call__(self, channel, *args, timeout):
        mode = args[0].split("=", 1)[0]
        if mode == "--paper-fetch":
            self.calls.append((channel, *args))
            return self.paper_fetch
        if mode == "--papers-pending":
            self.calls.append((channel, *args))
            return {"ok": True, "pending": self.papers}
        if mode == "--paper-register" and self.registered is not None:
            self.calls.append((channel, *args))
            return self.registered
        return await super().__call__(channel, *args, timeout=timeout)


def _paper_item(arxiv_id: str = "2609.04148") -> QueueItem:
    return QueueItem(id=7, source=Y.NAME, kind=Y.PAPER_KIND, priority=45,
                     payload={"arxiv_id": arxiv_id, "title": "Terminal-Universe", "max_turns": 30},
                     dedup_key=None, state="running", attempts=1, enqueued_at="",
                     claimed_at=None, claimed_by=None, completed_at=None, error=None)


def _paper_turn(text: str, *, note: Path | None = None, body: str | None = None, **kw):
    """A turn that, like the real one, replaces the placeholder with Write."""
    inner = _turn(text, **kw)

    async def run(prompt, **kwargs):
        if note is not None:
            note.write_text(body if body is not None else
                            PAPER_HEADER + "\n## Summary\n\n" + "finding " * 500
                            + "\n\n## Discussed in\n\n- [[20260920-qwen-learns]] — Discover AI, 20260920\n")
        out = await inner(prompt, **kwargs)
        run.prompt, run.kwargs = inner.prompt, inner.kwargs
        return out
    run.prompt, run.kwargs = "", {}
    return run


def test_the_block_names_papers_by_id_or_title_and_the_object_does_too():
    block = BLOCK.format(note="/n.md") + (
        "PAPERS: arXiv:2609.04148 | Is Your Model Thinking, or Just Stagnating? PUMA: Diagnosing\n"
        "Reasoning Pathology | `2607.21612` | four | five\n")
    parsed = Y.parse_result(block)
    assert parsed["papers"] == [
        "arXiv:2609.04148",
        "Is Your Model Thinking, or Just Stagnating? PUMA: Diagnosing Reasoning Pathology",
        "2607.21612", "four"], "a comma belongs to a title; at most MAX_PAPERS"
    assert parsed["filed"] == 523, "the fields above it are untouched"
    assert Y.parse_result(BLOCK.format(note="/n.md") + "PAPERS: none\n")["papers"] == []
    assert Y.parse_result(BLOCK.format(note="/n.md"))["papers"] == [], "a block from before the field"

    structured = {"result": "written", "note": "/n.md", "relevance": 50, "verdict": "background",
                  "areas": [], "source_kind": "paper", "approach": "read", "idea": "none",
                  "duplicate_of": "none", "filed": "none", "papers": "2609.04148 | A Title Of Some Words"}
    assert Y.parse_verdict("", structured)["papers"] == ["2609.04148", "A Title Of Some Words"]
    assert Y.RESULT_SCHEMA["properties"]["papers"]["maxLength"] == Y.DIGEST_PAPERS_MAX == 600
    assert "PAPERS:" in Y.TASK_BLOCK


async def test_the_papers_a_turn_names_are_registered_after_the_video_completes(
        tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])
    script = _PaperScript(registered={"ok": True, "registered": [{"arxiv_id": "2609.04148"}],
                                      "unresolved": ["A Title Nobody Wrote"]})
    script.fetch = {"ok": True, "meta": meta}
    text = BLOCK.format(note=note) + "PAPERS: 2609.04148 | A Title Nobody Wrote\n"
    _filed(backlog, 523)
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn(text, writes=note))

    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))

    assert result["status"] == "success", result
    # After --complete: the registry links each paper to the note that call records.
    assert script.modes() == ["--fetch", "--complete", "--eval-report", "--paper-register"]
    assert script.opt(3, "--paper-register") == "abc123", "attached, like every id this source passes"
    assert json.loads(script.opt(3, "--paper-refs")) == ["2609.04148", "A Title Nobody Wrote"]
    assert result["meta"]["papers_registered"] == ["2609.04148"]
    assert result["meta"]["papers_unresolved"] == ["A Title Nobody Wrote"]
    assert "1 paper(s) registered" in result["summary"]


async def test_a_registry_failure_does_not_cost_the_video_its_run(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])

    class Broken(_Script):
        async def __call__(self, channel, *args, timeout):
            if args[0].startswith("--paper-register"):
                self.calls.append((channel, *args))
                raise Y.ScriptError("--paper-register rc=1: arXiv unreachable")
            return await super().__call__(channel, *args, timeout=timeout)

    script = Broken({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session",
                        _turn(BLOCK.format(note=note) + "PAPERS: 2609.04148\n", writes=note))
    result = await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))
    assert result["status"] == "success" and "arXiv unreachable" in result["meta"]["papers_register_error"]


async def test_the_papers_switch_turns_off_both_halves(tmp_path, backlog, monkeypatch):
    meta = _meta(tmp_path)
    note = Path(meta["target_note"])
    off = {"papers": {"enabled": False}, "batch": 3, "channels": ["discover-ai"]}
    monkeypatch.setattr("workers.sources.get_sources_config", lambda: {Y.NAME: off})
    script = _PaperScript(papers=[{"arxiv_id": "2609.04148", "title": "T"}],
                          pending={"discover-ai": [{"video_id": "v1", "title": "V"}]})
    script.fetch = {"ok": True, "meta": meta}
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session",
                        _turn(BLOCK.format(note=note) + "PAPERS: 2609.04148\n", writes=note))
    await Y.execute(_item({"channel": "discover-ai", "video_id": "abc123"}))
    queue = WorkQueue(tmp_path / "workers.db")
    await Y.enqueue_if_due(queue, off)
    assert "--paper-register" not in script.modes() and "--papers-pending" not in script.modes()
    assert [i.kind for i in queue.list_items(source=Y.NAME)] == ["video"]


async def test_one_pending_paper_is_queued_ahead_of_the_videos(tmp_path, monkeypatch):
    queue = WorkQueue(tmp_path / "workers.db")
    script = _PaperScript(
        papers=[{"arxiv_id": "2609.04148", "title": "Terminal-Universe"},
                {"arxiv_id": "2607.21612", "title": "Second"}, {"title": "a row with no id"}],
        pending={"discover-ai": [{"video_id": f"v{i}", "title": f"V {i}"} for i in range(9)]})
    monkeypatch.setattr(Y, "_script", script)
    cfg = {"batch": 3, "channels": ["discover-ai"], "max_turns": 40, "priority": 45}

    await Y.enqueue_if_due(queue, cfg)
    items = queue.list_items(source=Y.NAME)
    papers = [i for i in items if i.kind == Y.PAPER_KIND]
    # One, not two: the channels always have a video pending, so a paper has to
    # go first to run at all — and one at a time, or a backfill of twenty
    # papers holds every slot the videos had.
    assert [p.payload["arxiv_id"] for p in papers] == ["2609.04148"]
    assert papers[0].dedup_key == "youtube-digest:paper:2609.04148" and papers[0].priority == 45
    assert sorted(i.payload["video_id"] for i in items if i.kind == "video") == ["v0", "v1"], "batch holds"

    # Still queued on the next tick: the same paper is offered and coalesces.
    await Y.enqueue_if_due(queue, {**cfg, "batch": 9})
    again = [i for i in queue.list_items(source=Y.NAME) if i.kind == Y.PAPER_KIND]
    assert [p.payload["arxiv_id"] for p in again] == ["2609.04148"]


async def test_a_paper_row_becomes_a_full_note(tmp_path, paper_skill, monkeypatch):
    meta = _paper_meta(tmp_path)
    note = Path(meta["target_note"])
    assert not Y._paper_note_is_real(note, "2609.04148"), "the placeholder is not the note"
    script = _PaperScript({"ok": True, "meta": meta})
    turn = _paper_turn(f"Read it.\n\nRESULT: written\nNOTE: {note}\n", note=note)
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", turn)

    result = await Y.execute(_paper_item())

    assert result["status"] == "success", result
    assert result["artifact_path"] == str(note) and result["summary"].startswith("Paper 2609.04148:")
    assert script.modes() == ["--paper-fetch", "--paper-complete"]
    assert script.opt(0, "--paper-fetch") == "2609.04148" and script.opt(1, "--paper-complete") == "2609.04148"
    assert result["meta"]["kind"] == "paper" and result["meta"]["session_id"]
    assert paper_skill == [Y.PAPER_SKILL] == ["paper-digest"]
    # The prompt: the vault skill, then every path and the header, verbatim —
    # a brace in a title is data, not a field.
    assert turn.prompt.startswith('[SYSTEM: You are running the "youtube-digest" worker job.')
    assert PAPER_SKILL_TEXT in turn.prompt and PAPER_HEADER in turn.prompt
    for needed in (meta["text_path"], meta["target_note"], meta["discussed"][0], str(Y.PROFILE_PATH),
                   "4500 words, 900 lines, from arXiv's HTML rendering", "RESULT: <written|failed>"):
        assert needed in turn.prompt, needed
    assert "size cap" not in turn.prompt
    # One note, nothing filed: the video's turn already judged the idea.
    assert set(Y.DISALLOWED) | {"backlog_write_task", "fact_add"} <= set(turn.kwargs["extra_disallowed"])
    assert turn.kwargs["max_turns"] == 30 and turn.kwargs["source"] == Y.NAME
    assert turn.kwargs["title"].startswith("Paper: Terminal-Universe")
    assert "final_schema" not in turn.kwargs


def test_every_paper_placeholder_is_supplied_and_a_cut_text_is_announced(tmp_path):
    meta = {**_paper_meta(tmp_path), "text_truncated": True, "text_source": "pdf"}
    fields = set(re.findall(r"\{(\w+)\}", Y.PAPER_TASK_BLOCK))
    prompt = Y.build_paper_prompt(meta, skill_text=PAPER_SKILL_TEXT)
    for name in sorted(fields):
        assert "{" + name + "}" not in prompt, f"{{{name}}} reached the prompt unrendered"
    assert "from arXiv's PDF" in prompt and "cut at its size cap" in prompt


@pytest.mark.parametrize("body, why", [
    (None, "the placeholder was left as it was"),
    (PAPER_HEADER + "\n## Summary\n\n" + "finding " * 500, "the write died before the last section"),
    (PAPER_HEADER.replace("2609.04148", "2601.00001") + "\n" + "x " * 2000 + "\n## Discussed in\n\n- a\n",
     "another paper's note"),
    (PAPER_HEADER + "\n## Summary\n\nShort.\n\n## Discussed in\n\n- a\n", "a header and nothing read"),
], ids=["untouched", "truncated", "wrong-paper", "too-short"])
async def test_a_paper_turn_without_a_full_note_is_a_counted_failure(tmp_path, paper_skill,
                                                                    monkeypatch, body, why):
    meta = _paper_meta(tmp_path)
    note = Path(meta["target_note"])
    script = _PaperScript({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session",
                        _paper_turn(f"RESULT: written\nNOTE: {note}\n",
                                    note=note if body is not None else None, body=body))
    result = await Y.execute(_paper_item())
    assert result["status"] == "failed", why
    assert script.modes() == ["--paper-fetch", "--paper-fail"], why
    assert "without a full note" in script.opt(1, "--reason")


async def test_a_paper_is_not_charged_for_a_drain_an_outage_or_a_missing_skill(
        tmp_path, paper_skill, monkeypatch):
    meta = _paper_meta(tmp_path)

    script = _PaperScript({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", raises=DrainActive("landing")))
    assert (await Y.execute(_paper_item()))["status"] == "skipped"
    assert script.modes() == ["--paper-fetch"]

    script = _PaperScript({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", stop_reason=None))
    result = await Y.execute(_paper_item())
    assert result["defer_seconds"] == Y.INFRA_DEFER_SECONDS and result["meta"]["infra"]
    assert script.modes() == ["--paper-fetch"], "an outage must not burn the paper's attempts"

    script = _PaperScript({"ok": True, "meta": meta})
    ran: list[str] = []

    async def never(prompt, **kwargs):
        ran.append(prompt)

    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", never)
    monkeypatch.setattr("app.autonomy._load_skill_content", lambda slug: "")
    result = await Y.execute(_paper_item())
    assert result["status"] == "failed" and result["meta"]["skill_missing"] and ran == []
    assert "skills/paper-digest/SKILL.md" in result["summary"]
    assert script.modes() == ["--paper-fetch"]

    # A timeout and a text that cannot be fetched are the paper's own.
    script = _PaperScript({"ok": True, "meta": meta})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr("app.autonomy._load_skill_content", lambda slug: PAPER_SKILL_TEXT)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", raises=TurnTimeout("1800s")))
    assert (await Y.execute(_paper_item()))["meta"]["turn_timeout"]
    assert script.modes() == ["--paper-fetch", "--paper-fail"]

    script = _PaperScript({"ok": False, "error": "Could not fetch paper text: html: 404", "failure_count": 2})
    monkeypatch.setattr(Y, "_script", script)
    result = await Y.execute(_paper_item())
    assert result["status"] == "failed" and result["meta"]["failure_count"] == 2
    assert script.modes() == ["--paper-fetch"], "the script already counted the attempt"
