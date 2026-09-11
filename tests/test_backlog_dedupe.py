"""`backlog_write_task` checks the board before it writes.

Triage was told to "run `backlog_tasks` to be sure no item already covers
it" — a tool with no text search that returns ~800 titles — and the loop
filed the same finding again every time a round re-ran: #549 ran four times
in 110 minutes and filed #788, #795 and #799 for one dead config floor.

The rules pinned here, in order of what would go wrong without them:

* a merge never loses text (it appends, under its own heading, naming the
  session), and a human's write is never merged, only advised;
* the reranker score alone cannot merge — an unrelated query still scored
  0.75 on its top hit — so the lexical leg must agree;
* the whole thing fails open: a daemon that is down costs the advisory list,
  never the write, and the result carries no `error` key (which is what
  `text_result` sniffs for `isError`).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as BL
from agent_mcp import backlog_similar as SIM


@pytest.fixture
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    monkeypatch.setattr(SIM, "DEDUPE_LOG", tmp_path / "dedupe.jsonl")
    monkeypatch.setattr(SIM, "dedupe_config", lambda: dict(SIM.DEFAULTS))
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: [])
    return d


def write(d: Path, item_id: int, name: str, body: str, *, status="draft",
          age_seconds: float = 3600.0, tags=("backlog",), broken_yaml=False) -> Path:
    created = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": "lloyd", "tags": list(tags)}
    text = yaml.dump(fm, default_flow_style=False)
    if broken_yaml:
        text += "tags: [unclosed\n"
    p = d / f"{item_id}-{name.lower()[:30].replace(' ', '-')}.md"
    p.write_text(f"---\n{text}---\n\n# {name}\n\n{body}\n", encoding="utf-8")
    return p


EXISTING = ("http_fetch fails on a quarter of its calls and its error body says only the status code",
            "`agent_mcp/http_tools.py:254` returns only `HTTP <n>`; 71 flagged errors over 268 calls.")
SAME = ("http_fetch error body says only the status code on a quarter of calls",
        "The error body from http_fetch carries only the status code; 71 errors over 268 calls "
        "in 21 days, and the model has to spend a turn deciding whether to retry.")


def _write(args: dict) -> dict:
    return json.loads(BL._handle_write(args))


def _spawn(name, description, **extra):
    return {"name": name, "description": description, "board": "lloyd",
            "tags": ["spawned-by-triage"], **extra}


def _vec(monkeypatch, rows):
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: rows)


# ── the merge ─────────────────────────────────────────────────────────────

def test_a_spawned_write_matching_an_open_item_is_merged_not_created(board, monkeypatch):
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    out = _write(_spawn(*SAME))
    assert out["success"] and out["created"] is False and out["merged_into"] == 10
    assert out["task_id"] == 10
    assert len(list(board.glob("*.md"))) == 1, "no second file"
    assert out["similar"][0]["id"] == 10


def test_a_merge_keeps_the_text_and_names_the_session(board, monkeypatch):
    from agent_mcp._task_registry import current_session_id
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    token = current_session_id.set("sess_triage_1")
    try:
        _write(_spawn(*SAME))
    finally:
        current_session_id.reset(token)
    text = next(board.glob("10-*.md")).read_text()
    assert "## Merged finding" in text
    assert SAME[0] in text and SAME[1] in text, "the would-be item's title and body both survive"
    assert "session sess_triage_1" in text
    fm = yaml.safe_load(text.split("---")[1])
    assert any("merged finding" in str(line) and "sess_triage_1" in str(line)
               for line in fm["activity_log"])
    assert "force: true" in text, "the reader is told how to undo a wrong merge"


def test_a_human_write_is_never_merged_only_advised(board, monkeypatch):
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.95}])
    out = _write({"name": SAME[0], "description": SAME[1], "board": "lloyd"})
    assert out["created"] is True and "merged_into" not in out
    assert [r["id"] for r in out["similar"]] == [10]
    assert len(list(board.glob("*.md"))) == 2


def test_force_bypasses_the_merge(board, monkeypatch):
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.95}])
    out = _write(_spawn(*SAME, force=True))
    assert out["created"] is True and out["similar"][0]["id"] == 10
    assert len(list(board.glob("*.md"))) == 2


@pytest.mark.parametrize("tag", ["umbrella", "blocker"])
def test_an_umbrella_or_blocker_write_is_never_merged(board, monkeypatch, tag):
    """An umbrella's description names the members it consolidates, so it
    matches every one of them; a blocker is the one item a round may file."""
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.95}])
    out = _write(_spawn(*SAME, tags=["spawned-by-triage", tag]))
    assert out["created"] is True


# ── what cannot merge ──────────────────────────────────────────────────────

def test_the_reranker_score_alone_cannot_merge(board, monkeypatch):
    """Measured: an unrelated query scored 0.75 on its top hit."""
    write(board, 10, "Rotate the guardian's voice log weekly",
          "voice.log grows without bound in the guardian state dir.")
    _vec(monkeypatch, [{"id": 10, "score": 0.97}])
    out = _write(_spawn(*SAME))
    assert out["created"] is True and "merged_into" not in out
    assert out["similar"] and out["similar"][0]["lexical"] < 0.4


def test_a_closed_match_is_advisory_not_a_target(board, monkeypatch):
    write(board, 10, *EXISTING, status="done")
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    out = _write(_spawn(*SAME))
    assert out["created"] is True and out["similar"][0]["status"] == "done"


def test_a_second_spawn_in_one_run_merges_lexically_before_qmd_has_seen_it(board, monkeypatch):
    """The watcher debounces 2 s and embedding takes longer; the only thing
    that matches a seconds-old item that strongly is the same session
    filing the same finding twice."""
    write(board, 10, *SAME, age_seconds=5, tags=("backlog", "spawned-by-triage"))
    _vec(monkeypatch, [])
    out = _write(_spawn(SAME[0], SAME[1] + " Also seen in session two."))
    assert out["merged_into"] == 10 and out["created"] is False


def test_an_old_item_needs_the_daemon_to_agree(board, monkeypatch):
    """Rule B is only for the recent window; an hour-old lexical twin with no
    reranker score is advised, not merged — that is rule A's job."""
    write(board, 10, *SAME, age_seconds=3600, tags=("backlog", "spawned-by-triage"))
    _vec(monkeypatch, [])
    out = _write(_spawn(SAME[0], SAME[1] + " Also seen in session two."))
    assert out["created"] is True and out["similar"][0]["id"] == 10


# ── failing open ───────────────────────────────────────────────────────────

def test_dedupe_fails_open_when_qmd_is_down(board, monkeypatch):
    write(board, 10, *EXISTING)
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: None)
    out = _write(_spawn(*SAME))
    assert out["success"] and out["created"] is True
    assert "error" not in out, "`text_result` sniffs a leading error key as isError"
    assert [r["id"] for r in out["similar"]] == [10], "the lexical leg still advises"


def test_a_raising_similarity_leg_never_blocks_the_write(board, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no daemon")
    monkeypatch.setattr(SIM, "similar_items", boom)
    out = _write(_spawn(*SAME))
    assert out["success"] and out["created"] is True and out["similar"] == []


def test_a_yaml_broken_target_falls_through_to_create(board, monkeypatch):
    """The writer refuses to round-trip a file it could only read by
    fallback; the finding must still land somewhere."""
    write(board, 10, *EXISTING, broken_yaml=True)
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    out = _write(_spawn(*SAME))
    assert out["created"] is True
    assert len(list(board.glob("*.md"))) == 2


# ── switches and schema ───────────────────────────────────────────────────

def test_dedupe_can_be_switched_off_and_set_advisory(board, monkeypatch):
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    monkeypatch.setattr(SIM, "dedupe_config", lambda: {**SIM.DEFAULTS, "merge": False})
    out = _write(_spawn(*SAME))
    assert out["created"] is True and out["similar"][0]["id"] == 10, "observation mode"
    monkeypatch.setattr(SIM, "dedupe_config", lambda: {**SIM.DEFAULTS, "enabled": False})
    out = _write(_spawn(SAME[0] + " again", SAME[1]))
    assert out["created"] is True and out["similar"] == []


def test_every_decision_is_logged_for_tuning(board, monkeypatch, tmp_path):
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    _write(_spawn(*SAME))
    rows = [json.loads(l) for l in (tmp_path / "dedupe.jsonl").read_text().splitlines()]
    assert rows[-1]["action"] == "merged" and rows[-1]["into"] == 10 and rows[-1]["rule"] == "A"


def test_the_schema_documents_force_and_a_create_reports_created():
    import asyncio
    tools = {t.name: t for t in asyncio.run(BL.list_tools())}
    props = tools["backlog_write_task"].input_schema["properties"]
    assert props["force"]["type"] == "boolean" and props["force"]["description"].strip()
    assert "merged_into" in tools["backlog_write_task"].description
    assert "similar" in tools["backlog_write_task"].description


def test_an_update_is_untouched_by_dedupe(board, monkeypatch):
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    out = _write({"task_id": 10, "description": "more", "description_mode": "append"})
    assert out["success"] and out["created"] is False and "similar" not in out
