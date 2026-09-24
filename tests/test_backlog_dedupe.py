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
          age_seconds: float = 3600.0, tags=("backlog",), broken_yaml=False,
          activity_log=()) -> Path:
    created = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": "lloyd", "tags": list(tags)}
    if activity_log:
        fm["activity_log"] = list(activity_log)
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


@pytest.mark.parametrize("switch", [True, False])
def test_a_youtube_eval_write_merges_like_a_spawn_unless_switched_off(board, monkeypatch, switch):
    """The digest files `youtube-eval` and no `spawned-by-*` tag, so a re-run
    of one video's evaluation filed it again as a human's item would be:
    advised, never merged. Loop output since 2026-09-14."""
    from app import config as CFG
    monkeypatch.setitem(CFG.CONFIG, "workers",
                        {"sources": {"youtube-digest": {"loop_spawned": switch}}})
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.95}])
    out = _write({"name": SAME[0], "description": SAME[1], "board": "lloyd",
                  "tags": ["youtube-eval", "ai-engineer"]})
    if switch:
        assert out.get("merged_into") == 10 and len(list(board.glob("*.md"))) == 1
    else:
        assert out["created"] is True and "merged_into" not in out


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


def test_a_closed_match_is_never_a_merge_target(board, monkeypatch):
    write(board, 10, *EXISTING, status="done")
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    out = _write(_spawn(*SAME))
    assert "merged_into" not in out and out["similar"][0]["status"] == "done"
    assert "Merged finding" not in next(board.glob("10-*.md")).read_text()


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


# ── #934: the token count is not a merge leg, and the head read reaches the status

# The 2026-09-14 instance: a knowledge-note frontmatter defect merged into
# #372 (log.md audit files) at score 1.0 / lexical 0.061 / shared 6 — the
# fleet's own vocabulary (okf, frontmatter, knowledge, segment) is what the
# two had in common. Different files, different fixes.
LOGMD = ("OKF Phase 2b: create log.md change history files",
         "Every knowledge note under the vault gets a log.md beside it recording each frontmatter "
         "edit with its session and date, so the okf validator can audit who changed a note's segment.")
SEGMENT = ("Knowledge-note writers emit files without a segment frontmatter key",
           "Three producers write knowledge notes with no segment in the frontmatter, and the okf "
           "validator flags every one of them on the next nightly pass; the writers need a default.")

# ~60 activity-log lines: the shape of an item that has been worked on, and
# the shape whose closing `---` sat past the old 2048-byte head read.
LONG_LOG = [f"**2026-09-{1 + i % 20:02d}T10:00:00** — triage pass {i} re-read the item and appended a "
            "finding about the retry path, the cooldown and the nightly window" for i in range(60)]


def _frontmatter_bytes(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return text.index("\n---\n", 3) + 5


def test_the_recorded_false_merge_now_creates():
    """Clause 1, on the recorded numbers: the row that merged on 09-14 is not
    a target however high the reranker and the token count read."""
    recorded = {"id": 372, "title": LOGMD[0], "status": "draft", "score": 1.0,
                "lexical": 0.061, "shared": 6, "created": 1_700_000_000.0, "source": "both"}
    assert SIM.merge_target([recorded], cfg=dict(SIM.DEFAULTS)) is None
    # The same row with the jaccard at the floor is still rule A, and a
    # seconds-old strong lexical twin is still rule B — the gate is not off.
    assert SIM.merge_target([{**recorded, "lexical": 0.4, "shared": 2}],
                            cfg=dict(SIM.DEFAULTS))["rule"] == "A"
    now = 1_700_000_100.0
    assert SIM.merge_target([{**recorded, "score": 0.0, "lexical": 0.6}],
                            cfg=dict(SIM.DEFAULTS), now=now)["rule"] == "B"


def test_shared_vocabulary_below_the_lexical_floor_is_advised_not_merged(board, monkeypatch):
    """Clauses 1 and 2 end to end: seven shared tokens at jaccard 0.17 against
    a reranker 1.0 creates, and the neighbour is still returned with its
    non-zero `shared` — a reported quantity, not a leg."""
    write(board, 372, *LOGMD)
    _vec(monkeypatch, [{"id": 372, "score": 1.0}])
    out = _write(_spawn(*SEGMENT))
    assert out["created"] is True and "merged_into" not in out
    assert len(list(board.glob("*.md"))) == 2
    row = out["similar"][0]
    assert row["id"] == 372 and row["score"] == 1.0
    assert row["shared"] >= SIM.DEFAULTS["shared_min"] and row["lexical"] < SIM.DEFAULTS["lexical_min"]


def test_a_closed_item_with_long_frontmatter_reads_closed_and_is_not_a_target(board, monkeypatch):
    """Clause 3: `status: done` past the old 2048-byte read is still `done`.
    The pair itself passes rule A (score 0.9, jaccard 0.46, the merge fixture
    above), so only the status keeps this from merging."""
    path = write(board, 10, *EXISTING, status="done", activity_log=LONG_LOG)
    assert _frontmatter_bytes(path) > 2048
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    out = _write(_spawn(*SAME))
    assert out["created"] is False and "merged_into" not in out
    assert out["similar"][0]["id"] == 10 and out["similar"][0]["status"] == "done"
    assert "Merged finding" not in path.read_text()


def test_a_long_frontmatter_item_reports_its_heading_and_created(board, monkeypatch):
    """Clause 4: the row carries the H1 as `title` and a non-null `created`,
    so the lexical leg compares prose to prose and rule B stays reachable —
    and an OPEN item of that shape is a rule A target again."""
    path = write(board, 10, *EXISTING, activity_log=LONG_LOG)
    assert _frontmatter_bytes(path) > 2048
    head = SIM._head(path, head_bytes=2048, body_chars=600)
    assert head["title"] == EXISTING[0] and head["status"] == "draft"
    assert head["created"] is not None and head["text"].startswith("`agent_mcp/http_tools.py:254`")
    assert "activity_log" not in head["text"], "the YAML is not the item's prose"
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    out = _write(_spawn(*SAME))
    assert out["merged_into"] == 10 and out["similar"][0]["title"] == EXISTING[0]


def test_a_dashed_line_in_the_prose_does_not_close_the_frontmatter(board):
    """`---` alone on a line inside the body is a horizontal rule, and a `---`
    inside a quoted activity-log entry is prose; only the anchored closing
    line ends the block."""
    log = ["**2026-09-12** — merged finding 'a --- b' from session x"]
    path = write(board, 10, EXISTING[0], "Intro paragraph.\n\n---\n\nSecond half.",
                 status="done", activity_log=log)
    head = SIM._head(path, head_bytes=2048, body_chars=600)
    assert head["status"] == "done" and head["title"] == EXISTING[0]
    assert head["text"].startswith("Intro paragraph.")



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


# ── #1051: a closed match explains itself, and a spawn does not re-file it ──

RETIRED_LOG = LONG_LOG + [
    "**2026-09-18T10:00:00.000000** — autotriage: **already_done**. `agent_mcp/http_tools.py:254` "
    "now returns the upstream body with the status. Check: `grep -n body agent_mcp/http_tools.py`"]


def _closed(board, item_id=10, **fm_extra):
    path = write(board, item_id, *EXISTING, status="done", activity_log=RETIRED_LOG)
    if fm_extra:
        text = path.read_text(encoding="utf-8")
        extra = yaml.dump(fm_extra, default_flow_style=False)
        path.write_text(text.replace("---\n", "---\n" + extra, 1), encoding="utf-8")
    assert _frontmatter_bytes(path) > 2048
    return path


def test_a_closed_row_carries_its_verdict_and_reason(board, monkeypatch):
    """Clause 3: `autotriage_retired` and the activity line's evidence, read
    through a front matter past the old window; `Check:` is the verifier's
    command, not the reason."""
    _closed(board, autotriage_retired="already_done")
    _vec(monkeypatch, [{"id": 10, "score": 0.5}])
    row = _write(_spawn(*SAME))["similar"][0]
    assert row["id"] == 10 and row["status"] == "done"
    assert row["verdict"] == "already_done"
    assert row["closure_reason"].startswith("`agent_mcp/http_tools.py:254` now returns")
    assert "Check:" not in row["closure_reason"]


def test_a_duplicate_close_names_its_survivor(board, monkeypatch):
    _closed(board, duplicate_of=7)
    _vec(monkeypatch, [{"id": 10, "score": 0.5}])
    row = _write(_spawn(*SAME))["similar"][0]
    assert row["verdict"] == "duplicate_of" and row["closure_reason"].startswith("duplicate of #7")


def test_a_close_with_no_recorded_verdict_says_so(board, monkeypatch):
    """Most `done` items were closed by a landing, expiry or a person and
    carry no verdict word: the row says that instead of guessing."""
    write(board, 10, *EXISTING, status="done", activity_log=LONG_LOG)
    _vec(monkeypatch, [{"id": 10, "score": 0.5}])
    row = _write(_spawn(*SAME))["similar"][0]
    assert row["verdict"] == SIM.NO_VERDICT and row["closure_reason"] == "closed, no recorded verdict"


def test_an_open_row_carries_no_closure_fields(board, monkeypatch):
    write(board, 10, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.5}])
    row = _write(_spawn(*SAME))["similar"][0]
    assert "verdict" not in row and "closure_reason" not in row


def test_a_spawn_matching_a_closed_item_is_refused_and_told_why(board, monkeypatch, tmp_path):
    """Clause 4: no file is written, and the result names the id, verdict,
    reason and the override — without a leading `error` key, which
    `text_result` would turn into a tool failure."""
    path = _closed(board, autotriage_retired="already_done")
    before = path.read_text(encoding="utf-8")
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    raw = BL._handle_write(_spawn(*SAME))
    out = json.loads(raw)
    assert out["refused"] is True and out["created"] is False and "task_id" not in out
    assert out["closed_match"]["id"] == 10 and out["closed_match"]["verdict"] == "already_done"
    assert out["closed_match"]["closure_reason"].startswith("`agent_mcp/http_tools.py:254`")
    assert "force: true" in out["message"] and "#10" in out["message"]
    assert "error" not in out
    assert [p.name for p in board.glob("*.md")] == [path.name]
    assert path.read_text(encoding="utf-8") == before
    rows = [json.loads(l) for l in (tmp_path / "dedupe.jsonl").read_text().splitlines()]
    assert rows[-1]["action"] == "refused_closed" and rows[-1]["match"] == 10


def test_force_creates_past_a_closed_match_and_names_it(board, monkeypatch):
    """Clause 5."""
    _closed(board, autotriage_retired="already_done")
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    out = _write(_spawn(*SAME, force=True))
    assert out["created"] is True and out["task_id"] == 11
    assert out["overrode_closed"]["id"] == 10 and out["overrode_closed"]["verdict"] == "already_done"
    assert "#10" in out["message"]
    assert len(list(board.glob("*.md"))) == 2


def test_a_human_write_or_a_weak_closed_match_is_only_advised(board, monkeypatch):
    """The refusal applies the merge rules to the loop's writes only: a
    person's create and a closed neighbour below rule A both create."""
    _closed(board, autotriage_retired="already_done")
    _vec(monkeypatch, [{"id": 10, "score": 0.9}])
    human = _write({"name": SAME[0], "description": SAME[1], "board": "lloyd"})
    assert human["created"] is True and "overrode_closed" not in human
    _vec(monkeypatch, [{"id": 10, "score": 0.5}])
    weak = _write(_spawn(SAME[0] + " again", SAME[1]))
    assert weak["created"] is True and "overrode_closed" not in weak


def test_an_open_match_still_merges_ahead_of_a_closed_one(board, monkeypatch):
    _closed(board, autotriage_retired="already_done")
    write(board, 12, *EXISTING)
    _vec(monkeypatch, [{"id": 10, "score": 0.9}, {"id": 12, "score": 0.9}])
    out = _write(_spawn(*SAME))
    assert out["merged_into"] == 12
