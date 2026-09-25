"""Post-compaction file restore reaches the engine (D3, 2026-09-24).

`restore_recent_files` used to return `role: "system"` rows, and
`_prepare_messages_for_harness` forwards only `user`/`assistant`/`tool` — so
every restored file was dropped on its way to the engine while `tokens_after`
still counted it, over-reporting the compacted size by up to the restore
budget (50k tokens). The restore is now ONE `user` row wrapped in
`<restored-context>`, and `tokens_after` counts only what is sent.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import app.compaction_llm as llm_mod
from app.compaction import estimate_conversation_tokens, load_and_compact_session
from app.compaction_llm import (
    RESTORED_CONTEXT_TAG,
    restore_recent_files,
    restored_file_count,
)
from app.routers._messages_harness_adapter import _prepare_messages_for_harness


def _read_turn(i: int, path: Path) -> list[dict]:
    cid = f"rd{i}"
    return [
        {"role": "user", "content": f"look at {path.name}"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": cid, "type": "function",
            "function": {"name": "Read", "arguments": json.dumps({"file_path": str(path)})},
        }]},
        {"role": "tool", "tool_call_id": cid, "content": "(old contents)"},
        {"role": "assistant", "content": "noted"},
    ]


def _files(tmp_path: Path, n: int) -> list[Path]:
    out = []
    for i in range(n):
        p = tmp_path / f"mod{i}.py"
        p.write_text(f"# module {i}\nVALUE = {i}\n" * 20)
        out.append(p)
    return out


def _text(row: dict) -> str:
    c = row.get("content")
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return str(c or "")


def test_restored_files_are_one_row_regardless_of_count(tmp_path):
    for n in (1, 4):
        d = tmp_path / str(n)
        d.mkdir()
        dropped = [m for i, p in enumerate(_files(d, n)) for m in _read_turn(i, p)]
        rows = restore_recent_files(dropped, max_files=5)
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["role"] == "user"
        text = _text(row)
        assert text.startswith(f"<{RESTORED_CONTEXT_TAG}>")
        assert text.rstrip().endswith(f"</{RESTORED_CONTEXT_TAG}>")
        assert text.count("<file path=") == n
        assert restored_file_count(rows) == n
    assert restore_recent_files([]) == []


def test_restored_file_count_does_not_read_file_contents(tmp_path):
    """A restored file may itself contain `<file` — the count rides on the row."""
    p = tmp_path / "tricky.md"
    p.write_text('<file path="x">not a real tag</file>\n' * 5)
    rows = restore_recent_files(_read_turn(0, p))
    assert restored_file_count(rows) == 1


def test_restored_context_rides_as_a_user_row_the_harness_adapter_keeps(tmp_path):
    files = _files(tmp_path, 3)
    dropped = [m for i, p in enumerate(files) for m in _read_turn(i, p)]
    rows = restore_recent_files(dropped)
    sent = asyncio.run(_prepare_messages_for_harness(
        [{"role": "assistant", "content": "[compaction summary]"}] + rows))
    assert [m["role"] for m in sent] == ["assistant", "user"], sent
    assert f"<{RESTORED_CONTEXT_TAG}>" in sent[1]["content"]
    for p in files:
        assert str(p) in sent[1]["content"]
    # The private count never reaches the wire.
    assert "restored_files" not in sent[1]


def test_tokens_after_counts_only_what_is_sent(tmp_path):
    """Summarize a history that read three files and carries one legacy
    `system` row (an old `/compact` restore): the restored files are in what
    is sent, and `tokens_after` is the estimate of exactly that."""
    files = _files(tmp_path, 3)
    msgs: list[dict] = [{"role": "system", "content": "[restored-context: file=/x]\n"
                         + "legacy " * 4_000}]
    for i, p in enumerate(files):
        msgs.extend(_read_turn(i, p))
    for _ in range(50):
        msgs.append({"role": "user", "content": "u" * 4_000})
        msgs.append({"role": "assistant", "content": "a" * 4_000})
    session = tmp_path / "s.json"
    session.write_text(json.dumps({"messages": msgs}))

    async def _summary(*_a, **_kw):
        return "the earlier work, summarized"

    saved = llm_mod.summarize_history
    llm_mod.summarize_history = _summary
    try:
        out = asyncio.run(load_and_compact_session(
            session, model="qwen-unknown", mode_override="summarize"))
    finally:
        llm_mod.summarize_history = saved

    assert out["summarized"] is True
    assert out["restored_files"] == 3, out["restored_files"]
    sent = asyncio.run(_prepare_messages_for_harness(out["history"]))
    restored = [m for m in sent if f"<{RESTORED_CONTEXT_TAG}>" in m["content"]]
    assert len(restored) == 1 and restored[0]["role"] == "user"
    expected = estimate_conversation_tokens(
        [m for m in out["history"] if m.get("role") in ("user", "assistant", "tool")],
        "",
    )
    assert out["tokens_after"] == expected
