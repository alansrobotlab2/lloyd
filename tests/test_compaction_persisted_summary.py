"""D2 (review 2026-09-24): the persisted, incremental, bounded summary.

`app/compaction_state.py` keeps the between-turn summary as a top-level
`data["compaction"]` record and folds new history into it instead of
regenerating it every turn. The summariser is monkeypatched to a counter here:
what is pinned is when it is called, with what, and what is stored — never
the text a model would write.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from app import compaction as C
from app import compaction_llm
from app import compaction_state as CS
from app.compaction import estimate_conversation_tokens, load_and_compact_session

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen-unknown"            # 128k default window → 76k threshold
ROW_CHARS = 8_000                 # ~2k tokens a row, ~4k a turn


def _run(coro):
    return asyncio.run(coro)


def _turn(i: int, chars: int = ROW_CHARS) -> list[dict]:
    tid = f"t{i:03d}"
    return [
        {"id": f"u{i:03d}", "role": "user", "turn_id": tid,
         "content": [{"type": "text", "text": f"U{i} " + "u" * chars}]},
        {"id": f"a{i:03d}", "role": "assistant",
         "content": [{"type": "text", "text": f"A{i} " + "a" * chars}]},
    ]


def _turns(lo: int, hi: int, chars: int = ROW_CHARS) -> list[dict]:
    out: list[dict] = []
    for i in range(lo, hi):
        out.extend(_turn(i, chars))
    return out


def _write(path: Path, messages: list[dict], **extra) -> None:
    data = {"session_id": path.stem, "last_active": "2026-09-24T10:00:00",
            "messages": messages}
    data.update(extra)
    path.write_text(json.dumps(data, indent=2))


def _data(path: Path) -> dict:
    return json.loads(path.read_text())


def _append(path: Path, rows: list[dict]) -> None:
    data = _data(path)
    data["messages"].extend(rows)
    path.write_text(json.dumps(data, indent=2))


class Summariser:
    """Stands in for `compaction_llm.summarize_incremental`."""

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    async def __call__(self, prior, delta, **kw):
        self.calls.append({"prior": prior, "delta": delta, **kw})
        if self.fail:
            return None
        n = len(self.calls)
        ids = ",".join(r.get("id", "") for r in delta)
        return (f"## Goal\nG{n}\n## Constraints\n-\n## Progress\nsaw {ids}\n"
                f"## Decisions\n-\n## Next steps\n-")


@pytest.fixture
def env(tmp_path, monkeypatch):
    from app.config import CONFIG
    from agent_mcp import _change_ledger as ledger
    from app import event_log

    cfg = {
        "mode": "summarize", "summary_model": None, "keep_recent_turns": 2,
        "persist_summary": True, "summary_input_budget_tokens": 200_000,
        "max_folds_per_turn": 3,
        "microcompact": {"enabled": False},
        "restore": {"enabled": False},
    }
    monkeypatch.setitem(CONFIG, "compaction", cfg)
    monkeypatch.setattr(event_log, "EVENT_LOGS_DIR", tmp_path / "event_logs")
    monkeypatch.setattr(ledger, "CHANGES_ROOT", tmp_path / "changes")
    ledger.reset()
    summariser = Summariser()
    monkeypatch.setattr(compaction_llm, "summarize_incremental", summariser)

    async def _legacy(*a, **k):  # the regenerate path must not run when on
        raise AssertionError("summarize_history called with persist_summary on")
    monkeypatch.setattr(compaction_llm, "summarize_history", _legacy)
    yield {"cfg": cfg, "path": tmp_path / "sess_d2.json", "sum": summariser,
           "events": tmp_path / "event_logs"}
    ledger.reset()


def _events(env, name: str) -> list[dict]:
    p = env["events"] / f"{env['path'].stem}.events.jsonl"
    if not p.exists():
        return []
    return [e for e in (json.loads(x) for x in p.read_text().splitlines() if x.strip())
            if e["event"] == name]


def _load(env) -> dict:
    return _run(load_and_compact_session(env["path"], model=MODEL))


# ---------------------------------------------------------------------------


def test_the_conversation_filter_is_one_definition():
    assert CS.CONVERSATION_ROLES is C._CONVERSATION_ROLES


def test_the_first_summary_is_persisted_with_its_boundary_and_sha(env):
    _write(env["path"], _turns(0, 30))
    out = _load(env)

    assert len(env["sum"].calls) == 1
    assert env["sum"].calls[0]["prior"] is None
    rec = _data(env["path"])["compaction"]
    convo = CS.conversation_rows(_data(env["path"])["messages"])
    # 30 turns, 2 kept recent → 28 turns (56 rows) covered.
    assert rec["covered_rows"] == 56
    assert rec["covers_through_index"] == 55
    assert rec["covers_through_entry_id"] == "a027"
    assert rec["covered_sha"] == CS.covered_sha(convo[:56])
    assert rec["folds"] == 1 and rec["source"] == "auto" and rec["version"] == 1
    assert rec["covered_turn_ids"][:2] == ["t000", "t001"]
    assert out["summarized"] is True and out["summarize_outcome"] == "summarized"
    assert out["summary_folds"] == 1 and out["summary_reused"] is False
    head = out["history"][0]
    assert head["role"] == "assistant"
    assert head["content"][0]["text"].startswith(CS.SUMMARY_HEADER)
    assert [m["id"] for m in out["history"][1:]] == ["u028", "a028", "u029", "a029"]


def test_a_later_turn_reuses_the_record_without_calling_the_summariser(env):
    _write(env["path"], _turns(0, 30))
    first = _load(env)
    _append(env["path"], _turn(30, 100))
    second = _load(env)

    assert len(env["sum"].calls) == 1, "a valid record is applied, not regenerated"
    assert second["summary_reused"] is True
    assert second["summarize_outcome"] == "reused"
    assert second["summary_folds"] == 0 and second["summary_covered_rows"] == 56
    # Byte-identical head: the prefix-cache win.
    assert second["history"][0] == first["history"][0]
    assert [m["id"] for m in second["history"][1:]][-2:] == ["u030", "a030"]


def test_a_record_applies_below_the_threshold_too(env):
    """The threshold decides folding, never whether the summary is used — a
    summary that came and went with the wall would re-prefill every time."""
    _write(env["path"], _turns(0, 30))
    _load(env)
    data = _data(env["path"])
    # Shrink what follows the boundary: same ids, tiny content.
    for m in data["messages"][56:]:
        m["content"] = [{"type": "text", "text": "x"}]
    env["path"].write_text(json.dumps(data))
    out = _load(env)
    assert out["tokens_before"] > 0
    assert out["summary_reused"] is True
    assert out["history"][0]["content"][0]["text"].startswith(CS.SUMMARY_HEADER)


def test_a_delta_past_the_boundary_is_folded_and_the_boundary_advances(env):
    _write(env["path"], _turns(0, 30))
    _load(env)
    first_summary = _data(env["path"])["compaction"]["summary"]
    _append(env["path"], _turns(30, 50))
    out = _load(env)

    assert len(env["sum"].calls) == 2
    call = env["sum"].calls[1]
    assert call["prior"] == first_summary, "folded INTO the prior summary"
    assert call["delta"][0]["id"] == "u028", "the delta starts at the old boundary"
    rec = _data(env["path"])["compaction"]
    # 22 turns past the boundary, 2 kept recent → 20 folded.
    assert rec["covered_rows"] == 96 and rec["covers_through_entry_id"] == "a047"
    assert rec["folds"] == 2
    assert out["summary_reused"] is True and out["summary_folds"] == 1


def test_a_rewritten_marker_does_not_invalidate_the_record(env):
    """`covered_sha` is over `id:role`: microcompact rewriting a covered
    result's content every load must not read as a changed past."""
    _write(env["path"], _turns(0, 30))
    _load(env)
    data = _data(env["path"])
    data["messages"][3]["content"] = [{"type": "text", "text": "[cleared]"}]
    env["path"].write_text(json.dumps(data))
    out = _load(env)
    assert out["summary_reused"] is True
    assert len(env["sum"].calls) == 1
    assert not _events(env, "compaction.record_invalidated")


def test_changed_covered_rows_discard_the_record_and_rebuild(env):
    _write(env["path"], _turns(0, 30))
    _load(env)
    data = _data(env["path"])
    # A whole-history rewrite (the legacy /compact) replaces the past.
    data["messages"] = _turns(100, 130)
    env["path"].write_text(json.dumps(data))
    out = _load(env)

    assert len(_events(env, "compaction.record_invalidated")) == 1
    assert len(env["sum"].calls) == 2
    assert env["sum"].calls[1]["prior"] is None, "rebuilt from scratch, not folded"
    assert out["summary_reused"] is False and out["summary_folds"] == 1
    assert _data(env["path"])["compaction"]["covers_through_entry_id"] == "a127"


def test_the_input_is_chunked_to_the_budget_and_progress_survives_the_fold_cap(env):
    env["cfg"]["summary_input_budget_tokens"] = 12_000
    env["cfg"]["max_folds_per_turn"] = 2
    _write(env["path"], _turns(0, 30))
    out = _load(env)

    calls = env["sum"].calls
    assert len(calls) == 2, "capped at max_folds_per_turn"
    for c in calls:
        assert estimate_conversation_tokens(c["delta"]) <= c["input_budget_tokens"]
        # A chunk is whole turns: it opens on a user row.
        assert c["delta"][0]["role"] == "user"
    assert calls[1]["prior"] is not None, "the second fold builds on the first"
    rec = _data(env["path"])["compaction"]
    end_of_second = calls[0]["delta"] + calls[1]["delta"]
    assert rec["covered_rows"] == len(end_of_second)
    assert rec["folds"] == 2
    # The leftover delta stays verbatim behind the summary...
    assert out["history"][0]["content"][0]["text"].startswith(CS.SUMMARY_HEADER)
    assert out["summarize_outcome"] == "summarized"

    # ...and the next turn keeps folding from where this one stopped.
    before = rec["covered_rows"]
    _load(env)
    assert len(env["sum"].calls) == 4
    assert env["sum"].calls[2]["delta"][0]["id"] == \
        CS.conversation_rows(_data(env["path"])["messages"])[before]["id"]
    assert _data(env["path"])["compaction"]["covered_rows"] > before


def test_each_fold_is_persisted_before_the_next_is_attempted(env, monkeypatch):
    env["cfg"]["summary_input_budget_tokens"] = 12_000
    seen: list[int] = []
    real = env["sum"]

    async def _spy(prior, delta, **kw):
        rec = _data(env["path"]).get("compaction")
        seen.append(rec["covered_rows"] if rec else 0)
        if len(seen) == 2:
            return None                      # the second fold fails
        return await real(prior, delta, **kw)

    monkeypatch.setattr(compaction_llm, "summarize_incremental", _spy)
    _write(env["path"], _turns(0, 30))
    out = _load(env)
    assert seen[0] == 0 and seen[1] > 0, "fold 1 was on disk before fold 2 ran"
    assert _data(env["path"])["compaction"]["covered_rows"] == seen[1]
    assert out["summary_folds"] == 1


def test_an_over_budget_turn_is_reduced_not_split():
    big = _turn(0, chars=80_000)
    chunks = CS.chunk_by_turns(_turns(1, 3) + big + _turns(3, 4), 10_000)
    ends = [end for end, _ in chunks]
    assert ends == [4, 6, 8], "breaks only on turn boundaries"
    reduced = chunks[1][1]
    assert estimate_conversation_tokens(reduced) <= 10_000 + 200
    assert [r["id"] for r in reduced] == ["u000", "a000"]


def test_files_touched_come_from_the_ledger_not_the_model(env):
    from agent_mcp import _change_ledger as ledger

    sid = env["path"].stem
    d = ledger.turn_dir(sid, "t003")
    d.mkdir(parents=True)
    (d / "index.json").write_text(json.dumps({"session_id": sid, "turn_id": "t003",
        "entries": [{"path": "/w/app/foo.py", "real": "/w/app/foo.py", "op": "edit"},
                    {"path": "/w/new.md", "real": "/w/new.md", "op": "create"}]}))
    _write(env["path"], _turns(0, 30))
    out = _load(env)

    rec = _data(env["path"])["compaction"]
    assert rec["files_touched"] == [{"path": "/w/app/foo.py", "op": "edit"},
                                    {"path": "/w/new.md", "op": "create"}]
    assert "Files touched" not in rec["summary"], "the model never writes it"
    text = out["history"][0]["content"][0]["text"]
    assert "## Files touched\n- /w/app/foo.py (edit)\n- /w/new.md (create)" in text
    assert "Do not list the files" in compaction_llm.INCREMENTAL_SYSTEM_PROMPT


def test_a_summariser_failure_leaves_the_prior_record_and_falls_back(env):
    _write(env["path"], _turns(0, 30))
    _load(env)
    rec_before = _data(env["path"])["compaction"]
    env["sum"].fail = True
    _append(env["path"], _turns(30, 60))
    out = _load(env)

    assert _data(env["path"])["compaction"] == rec_before
    assert out["summarize_outcome"] == "empty_summary"
    assert out["summary_reused"] is True and out["summarized"] is True
    # Still over the wall: truncation drops verbatim rows, never the summary.
    assert out["truncated"] is True
    assert out["history"][0]["content"][0]["text"].startswith(CS.SUMMARY_HEADER)


def test_a_first_summary_that_fails_writes_no_record_and_truncates(env):
    env["sum"].fail = True
    _write(env["path"], _turns(0, 30))
    out = _load(env)
    assert "compaction" not in _data(env["path"])
    assert out["summarized"] is False and out["summarize_outcome"] == "empty_summary"
    assert out["truncated"] is True


def test_summary_updated_is_logged_once_per_fold(env):
    _write(env["path"], _turns(0, 30))
    _load(env)
    _load(env)                               # reuse: no event
    ev = _events(env, "compaction.summary_updated")
    assert len(ev) == 1
    assert set(ev[0]["data"]) >= {"covers_through", "folds", "chars", "model",
                                  "duration_ms"}
    assert ev[0]["data"]["covers_through"] == 55


def test_the_record_is_written_after_messages_so_the_sweeps_prefix_read_finds_last_active(env):
    # A stale record placed FIRST (as no writer should) with a summary far
    # past 4 KB: the save must move it behind `messages`.
    stale = {"version": 1, "summary": "s" * 20_000}
    path = env["path"]
    data = {"compaction": stale, "session_id": path.stem,
            "last_active": "2026-09-01T10:00:00", "messages": _turns(0, 30)}
    path.write_text(json.dumps(data, indent=2))
    _load(env)

    keys = list(_data(path))
    assert keys.index("compaction") > keys.index("messages")
    spec = importlib.util.spec_from_file_location(
        "retention_sweep_d2", ROOT / "scripts/groundskeeper/retention-sweep.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    from datetime import datetime
    now = datetime.fromisoformat("2026-09-11T10:00:00").timestamp()
    assert rs._session_age_days(path, now) == pytest.approx(10.0)


def test_a_record_is_ignored_when_persist_summary_is_off(env, monkeypatch):
    """Off means today, exactly: the same history and the same result as a
    file with no record at all, through the regenerate path."""
    _write(env["path"], _turns(0, 30))
    _load(env)
    assert "compaction" in _data(env["path"])
    env["cfg"]["persist_summary"] = False
    legacy_calls: list[int] = []

    async def _legacy(older, **kw):
        legacy_calls.append(len(older))
        return "LEGACY"
    monkeypatch.setattr(compaction_llm, "summarize_history", _legacy)

    with_record = _load(env)
    bare = env["path"].with_name("bare.json")
    _write(bare, _data(env["path"])["messages"])
    without = _run(load_and_compact_session(bare, model=MODEL))

    assert with_record["history"] == without["history"]
    strip = lambda d: {k: v for k, v in d.items() if k != "history"}  # noqa: E731
    assert strip(with_record) == strip(without)
    assert with_record["summary_reused"] is False
    assert legacy_calls == [56, 56]
    assert len(env["sum"].calls) == 1, "only the first (on) turn folded"


def test_config_ships_the_keys_with_persist_summary_off():
    import yaml
    comp = yaml.safe_load((ROOT / "config.yaml").read_text())["compaction"]
    assert comp["persist_summary"] is False
    assert comp["summary_input_budget_tokens"] == 48_000
    assert comp["max_folds_per_turn"] == 3


def test_the_input_bound_respects_the_summary_models_window(monkeypatch):
    monkeypatch.setattr(C, "get_context_window", lambda m: 32_000)
    assert CS.input_budget({"summary_input_budget_tokens": 48_000}, "small") == \
        32_000 - CS.SUMMARY_MAX_OUTPUT_TOKENS - 2_000
    assert CS.input_budget({"summary_input_budget_tokens": 10_000}, "small",
                           "p" * 4_000) == 10_000 - 1_000


def test_summarize_incremental_clips_its_input_to_the_budget(monkeypatch):
    sent: list[dict] = []

    async def _post(**kw):
        sent.append(kw)
        return {"choices": [{"message": {"content": "<summary>## Goal\nx</summary>"}}]}
    monkeypatch.setattr(compaction_llm, "_summary_endpoint", lambda m: ("http://x", "m"))
    monkeypatch.setattr(compaction_llm, "_post_chat_completion", _post)
    rows = _turns(0, 10)
    out = _run(compaction_llm.summarize_incremental(
        "PRIOR", rows, input_budget_tokens=1_000))
    assert out == "## Goal\nx"
    user = sent[0]["messages"][1]["content"]
    assert "CURRENT SUMMARY:\nPRIOR" in user
    assert len(user) < 1_000 * 4 + 500
    assert "clipped to the summariser's input budget" in user
