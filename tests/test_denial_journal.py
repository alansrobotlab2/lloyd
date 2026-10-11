"""app/harness/denial_journal.py — one durable row per refusal, from every guard.

The claims pinned here: a PreToolUse deny is journaled under its gate's name
(a raising fail-closed gate included); the aggregator's `_refused_call` is
journaled under the guard its caller names; the session class is read off
the id; a journal that cannot be written never raises and never changes the
refusal; scorecard row 16 counts the file without importing the app; and the two
witness copies of quoted refusals under `backlog/data/` still measure what the
prose that quotes them says they measure (#2517).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.harness import denial_journal as DJ  # noqa: E402
from app.harness.hooks import HookRegistry  # noqa: E402


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "safety" / "denials.jsonl"
    monkeypatch.setenv(DJ.JOURNAL_ENV, str(path))
    return path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_a_hook_deny_is_journaled_under_its_gate(journal):
    async def _safety_pretool_cb(input_data, _id, _ctx):
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": "harness safety: blocked 'rm -rf on root/home/system path' on 'rm -rf ~'"}}
    hooks = HookRegistry()
    hooks.add_pre_tool_use("Bash", _safety_pretool_cb, fail_closed=True)
    out = _run(hooks.fire_pre_tool_use(session_id="20260930_120000_autocode_ab12",
                                       tool_name="Bash", tool_input={"command": "rm -rf ~"}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    rows = _rows(journal)
    assert len(rows) == 1
    r = rows[0]
    assert r["guard"] == "safety" and r["where"] == "hook" and r["tool"] == "Bash"
    assert r["session_class"] == "background"
    assert r["label"] == "rm -rf on root/home/system path"
    assert r["reason"].startswith("harness safety: blocked")
    assert "at" in r and "commit" in r


def test_a_raising_fail_closed_gate_is_journaled_too(journal):
    async def _policy_pretool_cb(input_data, _id, _ctx):
        raise RuntimeError("store unavailable")
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, _policy_pretool_cb, fail_closed=True)
    out = _run(hooks.fire_pre_tool_use(session_id="20260930_120000_chat_ab12",
                                       tool_name="email_send", tool_input={}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    rows = _rows(journal)
    assert len(rows) == 1 and rows[0]["guard"] == "grant" and rows[0]["tool"] == "email_send"
    assert "raised RuntimeError" in rows[0]["reason"]


def test_a_pass_and_a_fail_open_raise_write_nothing(journal):
    async def observer(input_data, _id, _ctx):
        raise RuntimeError("observer bug")

    async def passes(input_data, _id, _ctx):
        return {}
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, observer)
    hooks.add_pre_tool_use(None, passes)
    out = _run(hooks.fire_pre_tool_use(session_id="s", tool_name="Read", tool_input={}))
    assert out == {}
    assert _rows(journal) == []


def test_an_unlisted_gate_is_journaled_under_its_own_name(journal):
    async def my_custom_gate(input_data, _id, _ctx):
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": "no"}}
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, my_custom_gate)
    _run(hooks.fire_pre_tool_use(session_id="x", tool_name="Write", tool_input={}))
    assert _rows(journal)[0]["guard"].endswith("my_custom_gate")


def test_the_dispatch_refusal_helper_journals_under_the_named_guard(journal):
    from agent_mcp import main as M
    res = M._refused_call("Write", "read-only session: Write is not read-only",
                          guard="tool_sandbox", session_id="bench_588_x_abcd")
    assert res.is_error
    body = json.loads(res.content[0].text)
    assert body["error"].startswith("Tool call denied:")
    rows = _rows(journal)
    assert len(rows) == 1
    assert rows[0] == {**rows[0], "guard": "tool_sandbox", "where": "dispatch",
                       "tool": "Write", "session_class": "bench"}


@pytest.mark.parametrize("sid, cls", [
    ("", "none"),
    ("20260930_120000_autocode_ab12", "background"),
    ("20260930_120000_bench_ab12", "bench"),
    ("bench_588_key_ab12", "bench"),
    ("pt-eval-7", "bench"),
    ("task:abc", "subagent"),
    ("20260930_120000_ab12", "chat"),
    ("alan-chat", "chat"),
])
def test_session_class_is_read_off_the_id(sid, cls):
    assert DJ.session_class(sid) == cls


def test_an_unwritable_journal_never_raises_and_the_refusal_stands(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv(DJ.JOURNAL_ENV, str(blocker / "denials.jsonl"))  # parent is a file
    assert DJ.record(guard="safety", where="hook", session_id="s", tool="Bash",
                     reason="blocked") is False
    from agent_mcp import main as M
    res = M._refused_call("Bash", "harness safety: blocked 'x' on 'y'", guard="safety",
                          session_id="s")
    assert res.is_error


def test_rows_and_summarize_read_back_what_was_written(journal):
    DJ.record(guard="safety", where="hook", session_id="20260930_120000_autocode_1",
              tool="Bash", reason="harness safety: blocked 'a' on 'b'")
    DJ.record(guard="safety", where="dispatch", session_id="20260930_120000_1234",
              tool="Bash", reason="harness safety: blocked 'a' on 'c'")
    DJ.record(guard="grant", where="hook", session_id="20260930_120000_autocode_1",
              tool="email_send", reason="grant: no grant")
    rows = DJ.rows(journal)
    assert [r["guard"] for r in rows] == ["safety", "safety", "grant"]
    s = DJ.summarize(rows)
    assert s["total"] == 3
    assert s["by_guard"] == {"safety": 2, "grant": 1}
    assert s["by_class"] == {"background": 2, "chat": 1}
    assert s["by_where"] == {"hook": 2, "dispatch": 1}
    assert s["guard_by_class"]["safety"] == {"background": 1, "chat": 1}
    assert DJ.rows(journal.parent / "nope.jsonl") == []


def test_scorecard_row_16_counts_the_journal_without_the_app(journal, tmp_path, monkeypatch):
    import time
    from scripts.automod import scorecard as SC
    now = time.time()
    for i, (g, c) in enumerate([("safety", "background"), ("safety", "chat"),
                                ("tool_sandbox", "bench"), ("grant", "background")]):
        DJ.record(guard=g, where="hook" if i % 2 else "dispatch",
                  session_id={"background": "20260930_120000_autocode_1",
                              "chat": "20260930_120000_1234",
                              "bench": "bench_x_y_z"}[c],
                  tool="Bash", reason="harness safety: blocked 'lbl' on 'x'" if g == "safety" else "no")
    # one stale row, outside the window
    stale = {"at": "2020-01-01T00:00:00+00:00", "guard": "safety", "where": "hook",
             "session": "s", "session_class": "chat", "tool": "Bash", "label": "", "reason": "",
             "excerpt": "", "commit": ""}
    with journal.open("a") as fh:
        fh.write(json.dumps(stale) + "\n")
    ledger = tmp_path / "promotions.jsonl"
    ledger.write_text("")
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "nope",
                     repo=ROOT, now=now, denials=journal)
    d = row["guard_denials"]
    assert d["recorded"] is True and d["count"] == 4
    assert d["by_guard"] == {"safety": 2, "tool_sandbox": 1, "grant": 1}
    assert d["guard_by_class"]["safety"] == {"background": 1, "chat": 1}
    assert d["top_labels"] == {"safety:lbl": 2}
    text = SC.render(row)
    line = [l for l in text.splitlines() if l.startswith("| 16 |")][0]
    assert "safety 2 (" in line and "tool_sandbox 1 (bench 1)" in line
    # a missing journal is "not recorded", not zero
    row2 = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "nope",
                      repo=ROOT, now=now, denials=tmp_path / "absent.jsonl")
    assert row2["guard_denials"]["recorded"] is False
    assert "no journal yet" in SC.render(row2)
    # the default resolver honours the env override the writer honours
    assert SC._denial_journal_default() == journal


# ── the committed witnesses of quoted refusals (#2517) ───────────────────────
#
# `architecture/autonomy.md` quotes two refusals out of `<data root>/safety/denials.jsonl`
# and each quote names a copy under `backlog/data/` as its history. That is only true if the
# copy still measures what the prose says it measures, and until #2517 only one of the two
# copies existed: `backlog/data/denials.jsonl` is #2190's frozen extract, so the refusal
# #2362 quotes — 2026-10-05T06:38:16Z, `grant`, "dispatch-affecting field(s) [status]" on
# `target #96`, row 75 of the live journal — existed on one disk and in prose nowhere else
# (`git grep -c 'target #96' -- backlog/data/` was 0 across the whole committed store).
#
# Two copies, deliberately not one file. Appending #2362's denial to #2190's extract would
# have moved the 63 rows / 32,504 bytes #2190 re-derives from it, which is the reason the new
# witness is a separate path and the reason both sets of counts are pinned below rather than
# either one alone. Neither file is written by anything: they are extracts, and this node
# reads them, which is also why the change adds no write to `backlog/data/denials.jsonl`.
#
# The figures are measured, not transcribed:
#   wc -l -c < ~/obsidian/backlog/data/denials.jsonl              ->  63 32504
#   wc -l -c < ~/obsidian/backlog/data/denials.2362.witness.jsonl ->   1   678
#   sha256sum ~/obsidian/backlog/data/denials.2362.witness.jsonl
#     -> d24fada936c1dfb300dd3d6c2c2a26ed653af0307025cc80b474ff2fa7b9379e
#         (the extract-time digest of row 75 of the live journal, which the copy has kept)

#: #2190's frozen extract (`backlog/data/denials.jsonl`), at vault commit `3955c66c`.
_FROZEN_NAME = "denials.jsonl"
_FROZEN_ROWS = 63
_FROZEN_BYTES = 32504

#: #2362's denial (`backlog/data/denials.2362.witness.jsonl`), landed by #2517.
_WITNESS_NAME = "denials.2362.witness.jsonl"
_WITNESS_ROWS = 1
_WITNESS_BYTES = 678


def _vault_bytes(name: str) -> bytes:
    """One committed witness file's bytes, read from the vault the test asks about."""
    from tests.board_presence import vault_root

    path = vault_root() / "backlog" / "data" / name
    assert path.is_file(), (
        f"backlog/data/{name} is not on disk, so the refusal that quotes it has no committed "
        "bytes behind it. This node failing on a missing copy is the clause, not the "
        "environment — the copies are extracts, made once by hand and committed on the "
        "vault's main.")
    return path.read_bytes()


def _measures(raw: bytes) -> tuple[int, int]:
    """`(rows, bytes)` the way `wc -l -c` reports them: NUL-free newlines and size."""
    return len(raw.splitlines(keepends=True)), len(raw)


def test_the_committed_denials_witnesses_reproduce_their_quoted_reports():
    """#2517 clauses 1-3: the two witnesses measure what the prose quotes.

    The live journal is bounded by nothing (it is not one of the groundskeeper sweep's
    stores, and it has grown 63 → 203 rows since #2190's extract), so a quote that cited
    only `~/lloyd-data/safety/denials.jsonl` was a quote with an expiring denominator. The
    committed copies are the durable half of each citation, and the only thing that makes a
    copy durable is that something checks it.

    Both files are asserted twice over, and the second half is the point of the node: the
    counts alone say the copy is intact, not that it holds the row the paragraph is about.
    So #2362's row is re-read for the fields the quote names — its `at`, its `guard`, and
    both reason fragments — and #2190's extract is shown NOT to hold that target while
    holding a different one, which is the positive control that the absence is real and not
    a bad pattern or a wrong path.
    """
    raw = _vault_bytes(_WITNESS_NAME)
    assert _measures(raw) == (_WITNESS_ROWS, _WITNESS_BYTES), (
        f"the #2362 witness measures {_measures(raw)}, not "
        f"{_WITNESS_ROWS} row / {_WITNESS_BYTES} bytes. Truncating or appending to a frozen "
        "extract is exactly what this assertion exists to catch.")
    rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    assert len(rows) == _WITNESS_ROWS, rows
    row = rows[0]
    assert row["at"] == "2026-10-05T06:38:16+00:00", row["at"]
    assert row["guard"] == "grant", row["guard"]
    assert row["tool"] == "autonomy_write_task", row["tool"]
    assert row["where"] == "hook", row["where"]
    assert row["session"] == "20261004_232612_autocode_978b", row["session"]
    for fragment in ("dispatch-affecting field(s) [status]", "target #96"):
        assert fragment in row["reason"], (
            f"the committed denial does not say {fragment!r}, so the paragraph quoting it is "
            f"describing a row that is not in the file: {row['reason'][:200]!r}")

    frozen = _vault_bytes(_FROZEN_NAME)
    assert _measures(frozen) == (_FROZEN_ROWS, _FROZEN_BYTES), (
        f"#2190's frozen extract measures {_measures(frozen)}, not "
        f"{_FROZEN_ROWS} rows / {_FROZEN_BYTES} bytes — the figure #2190 quotes and the "
        "reason #2517 made a second file instead of growing this one")
    frozen_rows = [json.loads(ln) for ln in frozen.splitlines() if ln.strip()]
    assert len(frozen_rows) == _FROZEN_ROWS, len(frozen_rows)
    assert not [r for r in frozen_rows if "target #96" in r.get("reason", "")], (
        "#2190's extract now carries the #2362 denial, which means somebody appended to a "
        "frozen file and moved the count above as well")
    # Positive control: the 0 above is an answer, not a dead pattern or a wrong path. The
    # same file has one `target #new` row and two `guard: grant` rows.
    assert sum("target #new" in r.get("reason", "") for r in frozen_rows) == 1, (
        "the frozen extract's own target-bearing row is gone, so the absence check above "
        "was reading an empty or unrelated file")
    assert sum(1 for r in frozen_rows if r.get("guard") == "grant") == 2, (
        f"the frozen extract holds {sum(1 for r in frozen_rows if r.get('guard') == 'grant')} "
        "grant rows, not the two #2190's commit message counts")
    assert max(r["at"] for r in frozen_rows) == "2026-10-04T18:23:51+00:00", (
        "the extract stopped being frozen: its last row is the date the paragraph gives for "
        "why #2362's denial cannot be inside it")


def test_the_witness_count_check_rolls_over_on_a_quiet_truncation(tmp_path):
    """The node above can fail, and fails for both copies, on bytes this file writes.

    A pin that reads a committed file has to be shown red somewhere, or a moved path and a
    truncated store look the same as a clean pass. Both directions of the one accident this
    node guards against are driven here — a lost final row, and a row whose tail went — in a
    scratch directory, so the committed bytes are never the thing under test twice.
    """
    committed = _vault_bytes(_WITNESS_NAME)
    assert _measures(committed) == (_WITNESS_ROWS, _WITNESS_BYTES)

    dropped = b"".join(committed.splitlines(keepends=True)[:-1])
    assert _measures(dropped) != (_WITNESS_ROWS, _WITNESS_BYTES), (
        "a witness with its only row removed still measured the pinned count, so the count "
        "assertion proves nothing")
    clipped = committed[: _WITNESS_BYTES - 40]
    assert _measures(clipped) != (_WITNESS_ROWS, _WITNESS_BYTES), (
        "a witness with the tail of its row clipped still measured the pinned byte count")
    grown = _vault_bytes(_FROZEN_NAME) + b'{"at": "2026-10-05T06:38:16+00:00"}\n'
    assert _measures(grown) != (_FROZEN_ROWS, _FROZEN_BYTES), (
        "appending #2362's denial to #2190's extract left the frozen count alone, so the "
        "frozen assertion proves nothing")
    for name, bad in ((_WITNESS_NAME, dropped), (_WITNESS_NAME, clipped),
                      (_FROZEN_NAME, grown)):
        victim = tmp_path / name
        victim.write_bytes(bad)
        assert victim.read_bytes() == bad, "the scratch fixture did not persist"
