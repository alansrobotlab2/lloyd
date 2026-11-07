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
import subprocess
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


def test_a_hook_side_denial_row_names_the_command_it_refused(journal):
    """#2801 clause 5, across the one boundary that used to lose the command.

    The process boundary is the hook contract itself: `safety._safety_pretool_cb`
    builds a `(label, excerpt)` pair — with an 80-character command excerpt for the
    live-checkout git lane — and then flattens it into a single
    `permissionDecisionReason` string, because that is all a PreToolUse deny may
    carry. `hooks.fire_pre_tool_use` sees that string and nothing else, and
    `record()` caps `reason` at 400 characters. So on this route the command was
    structurally unrecoverable: all 50 live-checkout rows in
    `~/lloyd-data/safety/denials.jsonl` for 2026-10-12..16 have `excerpt: ""`, which
    is what made #2604's owed ruling ("did the lane fire only on its failure mode?")
    unanswerable without opening transcripts. The fix carries the refused call's
    `tool_input` alongside the deny, which is what this node crosses.

    The label half is the other half of the same blindness: `_SAFETY_LABEL` read only
    a single-quoted pattern, and this lane's message contains an apostrophe ("a turn
    commits in its round's worktree"), so `repr` quotes it with double quotes and the
    label came out empty too.
    """
    from app.harness import safety

    live = str(__import__("app.paths", fromlist=["LIVE"]).LIVE_CHECKOUT)
    cmd = f"git -C {live} commit -m x -- architecture/foo.md"
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, safety._safety_pretool_cb, fail_closed=True)
    out = _run(hooks.fire_pre_tool_use(session_id="20261016_090000_autocode_2801",
                                       tool_name="Bash",
                                       tool_input={"command": cmd,
                                                   "cwd": "/tmp/scratch"}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    rows = _rows(journal)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["guard"] == "safety" and row["where"] == "hook", row
    # The refused command, whole, and the lane that refused it — the two fields a
    # later pass needs to classify a firing without a transcript.
    assert row["excerpt"] == cmd, row
    assert row["label"].startswith("live-checkout git write"), row["label"]
    assert len(row["label"]) <= DJ.LABEL_CHARS, row["label"]


def test_the_excerpt_and_label_helpers_read_what_the_rows_actually_carry():
    """The two helpers behind clause 5, pinned at the unit level for the shapes the
    fleet's rows contain rather than the one the end-to-end node happens to fire.

    `excerpt_of` prefers the field that *is* the request — a Bash command, a file
    path — over dumping the whole argument dict, because the dict of a `Write` carries
    its content and the useful 200 characters would be the first 200 of a file. A
    multi-line command is folded to one line because the row is one line of JSONL and
    the classifier greps it; the 200-character cap is `record`'s, and the long arm
    here is what shows the cap is a truncation and not a drop.
    """
    assert DJ.excerpt_of("Bash", {"command": "git status\ngit add -A\n",
                                  "cwd": "/tmp"}) == "git status git add -A"
    assert DJ.excerpt_of("Write", {"file_path": "/srv/prod/lloyd/app/x.py",
                                   "content": "x" * 500}) \
        == "/srv/prod/lloyd/app/x.py"
    assert DJ.excerpt_of("Bash", {"command": "y" * 400})[:200] == "y" * 200
    assert DJ.excerpt_of("Bash", "not a dict") == ""
    assert DJ.excerpt_of("Bash", {}) != "", "an empty dict still gets the JSON form"

    # Both quote styles `repr` may have chosen, and the absence of either.
    assert DJ.label_of("harness safety: blocked 'rm -rf ~' on 'rm -rf ~'") \
        == "rm -rf ~"
    assert DJ.label_of('harness safety: blocked "live-checkout git write: `git add` '
                       'in a round\'s worktree" on `git add f`') \
        == "live-checkout git write: `git add` in a round's worktree"
    assert DJ.label_of("grant: no grant for that") == ""


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


# ---------- #3874: a refused command survives the cut that dropped its target -----

#: The row shape `record()` wrote before #3874, asserted verbatim by the two nodes
#: that pin what must NOT change. Live rows carry exactly these ten keys:
#: `tail -1 ~/lloyd-data/safety/denials.jsonl | jq 'keys | sort'`.
_TODAYS_KEYS = ["at", "commit", "excerpt", "guard", "label", "reason",
                "session", "session_class", "tool", "where"]


def _live_checkout_commit_command() -> str:
    """A refused `git commit` in the live checkout long enough that its target flags
    fall past `EXCERPT_CHARS`.

    The padding is the point: the flags are what identifies WHICH checkout, and the
    live journal proves the cut is where they are lost — `grep -c -- '--git-dir'` over
    `~/lloyd-data/safety/denials.jsonl` is 0 across its 981 rows (measured 2026-11-07,
    the whole file, spanning 2026-10-08 to 2026-11-07), and 326 of those rows sit
    exactly at the 200-char cap, every one of them a `Bash` row. Whitespace-folded on
    purpose: a heredoc is what the real refusals look like (`app.harness.safety._fold`
    is why the excerpt of a multi-line command fits in 200 characters at all), and the
    offsets this returns are the offsets `excerpt_of` will fold to.
    """
    from app.paths import LIVE_CHECKOUT
    pad = "explain the why " + ("x" * 300)
    return ('git -C %s commit -m "%s" \\\n'
            '  --git-dir=%s/.git --work-tree=%s -- architecture/foo.md'
            % (LIVE_CHECKOUT, pad, LIVE_CHECKOUT, LIVE_CHECKOUT))


def _refused_by_the_real_lane(command: str, session: str = "20261103_000000_chat") -> dict:
    """Refuse `command` through `app.harness.safety`'s own PreToolUse callback — the
    gate, not a stub — and return its deny decision."""
    from app.harness import safety

    async def _deny_cb(input_data, _id, _ctx):
        return await safety._safety_pretool_cb(input_data, _id, _ctx)

    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, _deny_cb, fail_closed=True)
    out = asyncio.new_event_loop().run_until_complete(
        hooks.fire_pre_tool_use(session_id=session, tool_name="Bash",
                                tool_input={"command": command}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny", (
        "the fixture command is no longer refused by the safety lane, so these nodes "
        "would be measuring a refusal that cannot happen")
    return out


def test_a_refused_command_past_the_cut_keeps_the_target_it_was_refused_for(journal):
    """#3874 clause 1: the command is carried whole up to `FULL_CMD_CHARS`, so a
    `--git-dir=` / `--work-tree=` flag sitting past character 200 is in the row.

    Refused by `app.harness.safety` itself through the registry, not by a stub, so the
    row is one the writer will really write.

    A test fixture, deliberately: the flag offset is what the clause pins, and the
    #3874 triage found no refused `git --git-dir=` invocation in the newest 1,155
    session files (9 commands name those flags at all, 4 with the flag past the cut, at
    offsets 372 / 603 / 799 / 1020, all of them heredocs mentioning them). So the live
    journal's 0 cannot be made non-zero by any change to the writer, and >= 1 is a
    fixture measurement — which is what clause 2 says to read it as.
    """
    cmd = _live_checkout_commit_command()
    folded = " ".join(cmd.split())
    offset = folded.index("--git-dir=")
    assert offset > DJ.EXCERPT_CHARS, (
        f"the fixture's flag sits at {offset}, inside the cut — it could never show "
        f"what #3874 is about")
    assert "--git-dir" not in folded[:DJ.EXCERPT_CHARS], "the flag is not past the cut"

    _refused_by_the_real_lane(cmd)

    row = _rows(journal)[0]
    assert row["full_command"], "the long refused command was not carried at all"
    assert "--git-dir=" in row["full_command"] and "--work-tree=" in row["full_command"], (
        "the refused command lost the flags that name which checkout it targeted")
    assert row["full_command"].index("--git-dir=") == offset, (
        "the carried command is not the folded command, so its offsets are not the "
        "ones a reader would quote")
    assert len(row["full_command"]) <= DJ.FULL_CMD_CHARS == 1200, (
        f"the field ran to {len(row['full_command'])} chars against the "
        f"{DJ.FULL_CMD_CHARS}-char bound")


def test_the_journal_answers_the_git_dir_grep_a_window_claim_needs(journal):
    """#3874 clause 2: `grep -c -- '--git-dir'` over the rows this writes is >= 1 —
    the measurement that is 0 over the live journal today — while `excerpt` is still
    exactly what `excerpt_of` produces, byte for byte.

    The grep is run by the same tool the claim is quoted with, over the same file,
    because the whole point of the item is that window claims are made with grep.
    """
    cmd = _live_checkout_commit_command()
    _refused_by_the_real_lane(cmd, session="20261103_000001_chat")

    counted = subprocess.run(["grep", "-c", "--", "--git-dir", journal],
                             capture_output=True, text=True)
    assert int(counted.stdout.strip() or 0) >= 1, (
        "grep -c -- '--git-dir' still finds nothing in the journal the fix writes, "
        "which is the 0 the item was filed over")

    row = _rows(journal)[0]
    assert row["excerpt"] == DJ.excerpt_of("Bash", {"command": cmd})[:DJ.EXCERPT_CHARS]
    assert len(row["excerpt"]) == DJ.EXCERPT_CHARS, (
        f"excerpt is {len(row['excerpt'])} chars, not today's {DJ.EXCERPT_CHARS}-char cut")
    assert "--git-dir" not in row["excerpt"], (
        "the excerpt grew, so the row no longer shows what a row written before #3874 "
        "looks like — the size that clause 3 bounds is this one")


async def _denies_whatever_it_is_asked_about(input_data, _id, _ctx):
    """A gate that refuses anything, for the nodes that are about the ROW and not about
    which pattern matched. Clause 3's 20 KB command is not a shape `app.harness.safety`
    has a lane for, so pinning the bound needs a refusal that does not depend on one."""
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": "guard: blocked x on x"}}


def test_a_twenty_kilobyte_command_is_capped_and_the_row_stays_one_line(journal):
    """#3874 clause 3: at 20 KB the field contributes at most `FULL_CMD_CHARS`, is
    whitespace-folded like the excerpt, and the row is still a single line whose total
    length is the field plus the row's own overhead and nothing else.

    The overhead is measured, not assumed: the same guard refusing a short command
    writes no field at all, so the difference between the two lines IS the cost.
    """
    huge = "echo start " + ("a\n\t b " * 5000)          # ~20 KB of foldable filler
    assert len(huge) > 20_000, f"the fixture is {len(huge)} chars, not 20 KB"
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, _denies_whatever_it_is_asked_about, fail_closed=True)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(hooks.fire_pre_tool_use(
        session_id="20261103_000002_chat", tool_name="Bash",
        tool_input={"command": huge}))
    loop.run_until_complete(hooks.fire_pre_tool_use(
        session_id="20261103_000003_chat", tool_name="Bash",
        tool_input={"command": "git status"}))

    lines = [ln for ln in journal.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2, f"{len(lines)} rows for 2 refusals: the row is not one line"
    long_row, short_row = (json.loads(ln) for ln in lines)
    field = long_row["full_command"]
    assert len(field) == DJ.FULL_CMD_CHARS, (
        f"a 20 KB command wrote {len(field)} chars, not the {DJ.FULL_CMD_CHARS}-char cap")
    assert field == " ".join(field.split()), "the field is not whitespace-folded"
    assert "\n" not in field and "\t" not in field and "\\" not in field, (
        "the field carries a character that could break the one-row-per-line shape")

    # The overhead is this same row without its field, re-serialised the way `record()`
    # writes it — NOT a different refusal's line, whose `excerpt` and `reason` are their
    # own lengths and would be counted as if they were the field's cost.
    baseline = json.dumps({k: v for k, v in long_row.items() if k != "full_command"},
                          sort_keys=True)
    escapes = len(json.dumps(field)) - len(field)   # bytes the JSON encoding adds
    cost = len(lines[0]) - len(baseline)
    assert cost <= DJ.FULL_CMD_CHARS + escapes + len(',"full_command":""'), (
        f"the field costs {cost} B on the line against a {DJ.FULL_CMD_CHARS}-char bound "
        f"(+{escapes} B of encoding), so the bound is not what is stored")
    assert len(lines[0]) < len(huge), "the row kept a 20 KB command whole"
    assert sorted(short_row) == _TODAYS_KEYS, (
        f"a short command now writes {sorted(short_row)} — a row <=EXCERPT_CHARS is "
        "supposed to keep exactly today's key set, which is the baseline clause 5 pins")


def test_a_content_lane_refusal_carries_no_more_of_its_payload_than_today(journal):
    """#3874 clause 4: the field is populated ONLY from a request's `command` key.

    The write and outbound-content lanes hand the journal a `redact()`-ed excerpt
    (`app/harness/outbound_content.py:604`), and `excerpt_of` already pulls their
    `content` / `text` / `prompt` / `url` bodies into the row unredacted up to 200
    chars. A whole-request field on THOSE lanes would widen how much of a
    secret-bearing payload lands in a file that outlives the payload, so the
    exposure is bounded by scope, not by redaction parity: a 20 KB `Write` body and
    a 20 KB `http_request` body must leave the row exactly as long as it is today.

    Both seams that can reach the writer are exercised, because they are different
    processes' code: the PreToolUse hook seam (`app/harness/hooks.py`) and the
    dispatch seam (`agent_mcp/main.py::_refused_call`). Measured over the live journal
    on 2026-11-07 the two routes carry 452 `safety`/`hook`/`Bash` rows and 406
    `tool_sandbox`/`dispatch`/`Write` rows between them, so both are load-bearing and
    neither may grow a payload body."""
    secret = "hunter2-never-journal-this " * 1000        # ~23 KB of body
    assert len(secret) > 20_000, f"the payload is {len(secret)} chars, not 20 KB"

    # Seam 1: the PreToolUse hook route, which is where the 452 live `safety`/Bash rows
    # come from and where `tool_input` is available in full.
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, _denies_whatever_it_is_asked_about, fail_closed=True)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(hooks.fire_pre_tool_use(
        session_id="20261103_000004_chat", tool_name="Write",
        tool_input={"file_path": "/home/alansrobotlab/obsidian/x.md", "content": secret}))
    loop.run_until_complete(hooks.fire_pre_tool_use(
        session_id="20261103_000005_chat", tool_name="http_request",
        tool_input={"url": "https://example.invalid/tok/" + secret, "method": "POST",
                    "body": secret}))
    # Seam 2: the aggregator's dispatch route, across the same boundary the fix had to
    # cross — a Bash refusal there MUST carry the command, a Write refusal there must
    # not carry the body.
    from agent_mcp.main import _refused_call

    _refused_call("Bash", "harness safety: blocked 'sudo' on 'sudo rm -rf /var…'",
                  guard="safety", session_id="20261103_000006_chat",
                  arguments={"command": "sudo rm -rf /var/lib " + ("y" * 4000),
                             "timeout": 120})
    _refused_call("Write", "read-only session: this session may not write",
                  guard="tool_sandbox", session_id="20261103_000007_chat",
                  arguments={"file_path": "/home/alansrobotlab/obsidian/y.md",
                             "content": secret})
    # Seam 3: `outbound_content.py:604`'s own shape — an excerpt and nothing else.
    DJ.record(guard="outbound_content", where="dispatch",
              session_id="20261103_000008_chat", tool="email_send",
              excerpt="outbound content: blocked 'secret-read' on " + secret,
              reason="outbound content: blocked 'secret-read' on a 20 KB body")

    rows = _rows(journal)
    assert len(rows) == 5, f"{len(rows)} rows for 5 refusals"
    bash = [r for r in rows if r["tool"] == "Bash"]
    assert len(bash) == 1 and "--git-dir" not in bash[0].get("full_command", "x")
    assert bash[0]["full_command"].startswith("sudo rm -rf /var/lib")
    assert len(bash[0]["full_command"]) == DJ.FULL_CMD_CHARS, (
        "the dispatch seam did not apply the cap")
    for row in rows:
        if row["tool"] == "Bash":
            continue
        assert "full_command" not in row, (
            f"the {row['tool']} lane grew a whole-payload field: this item is scoped to "
            "the `command` key precisely so a secret stays 200 characters deep")
        assert len(row["excerpt"]) <= DJ.EXCERPT_CHARS, (
            f"a content-lane excerpt is {len(row['excerpt'])} chars, past today's "
            f"{DJ.EXCERPT_CHARS}-char cut")
        assert secret not in json.dumps(row), (
            f"a {len(secret)}-char payload body reached the durable row whole on the "
            f"{row['tool']} lane")


def test_a_command_within_the_cut_writes_exactly_todays_key_set(journal):
    """#3874 clause 5: a refusal whose folded command is <= `EXCERPT_CHARS` is written
    with today's ten keys and no eleventh, so every existing reader of the journal —
    `scripts/automod/scorecard.py::_denials`, `rows()` and `summarize()`, and the
    committed-witness nodes in this file — keeps reading rows it never heard of.

    The short/long boundary is where a naive implementation leaks cost: writing the
    field whenever a `command` exists would put it on the majority of live rows (most
    refusals in the journal are one-line destructive-verb shapes, not long heredocs)
    and re-base store 26's fold sizing for nothing.
    """
    _refused_by_the_real_lane("git -C /home/alansrobotlab/lloyd commit -m x",
                              session="20261103_000007_chat")

    rows = _rows(journal)
    assert len(rows) == 1, "the short live-checkout commit was not journalled once"
    assert sorted(rows[0]) == _TODAYS_KEYS, (
        f"a <=EXCERPT_CHARS refusal now writes {sorted(rows[0])} — scorecard row 16 and "
        "the witness nodes read this key set")
    assert rows[0]["excerpt"] == "git -C /home/alansrobotlab/lloyd commit -m x"
    assert json.loads(json.dumps(rows[0])) == rows[0], "the row is not round-trippable JSON"


def test_the_dispatch_surface_keeps_todays_keys_short_and_the_tail_long(journal):
    """#3874 clause 5, the SECOND write surface: `agent_mcp/main.py::_refused_call` hands
    the writer a `command` and no `excerpt` at all, so the shape rule cannot rest on a
    comparison against the row's own excerpt — on that surface every non-empty command
    would look "longer than what the row kept" and a 15-char refusal would gain an
    eleventh key carrying text the row already had.

    Both directions are pinned through the aggregator's own function, not the hook lane: a
    short command refused by the bench durable-write rail keeps exactly today's ten keys
    with the `excerpt` that lane has always left empty, and a long one refused by the
    dispatch safety lane carries `full_command` with the flag that lane refuses on. That
    lane is not a corner: the live journal holds 36 `safety`/`dispatch`/`Bash` rows.
    """
    from agent_mcp.main import _refused_call

    _refused_call("Bash", "write outside the workspace is refused for this session",
                  guard="bench_durable_write", session_id="bench_short_3874",
                  arguments={"command": "rm -rf ./build", "summary": "delete build"})
    _refused_call("Bash", "harness safety: blocked 'live-checkout-git' on 'git --git-dir=…'",
                  guard="safety", session_id="20261103_000008_chat",
                  arguments={"command": _live_checkout_commit_command()})

    short, long_row = _rows(journal)
    assert sorted(short) == _TODAYS_KEYS, (
        f"a 15-char command refused at dispatch now writes {sorted(short)} — scorecard row "
        "16 and the witness nodes read this key set")
    assert short["excerpt"] == "", "the dispatch lane wrote no excerpt before #3874, and none now"
    assert long_row["full_command"].startswith("git ") and "--git-dir=" in long_row["full_command"], (
        "the dispatch safety lane still loses the target its own refusal named")
    assert long_row["excerpt"] == "", "wiring `arguments` must not start writing this lane's excerpt"


def test_the_full_command_constant_is_1200_and_is_what_the_writer_uses():
    """#3874's sizing decision, pinned as a number: `FULL_CMD_CHARS = 1200`, not the 4000
    the item suggested.

    Store 26 of the retention sweep (#3789, `scripts/groundskeeper/retention-sweep.py`)
    copies this file whole into `backlog/data/` up to `WITNESS_MAX_BYTES = 1_048_576`
    (`scripts/automod/backlog.py:1450`), and its comment sizes the headroom on the
    largest rolling 30-day volume observed at landing. Re-measured on 2026-11-07 over a
    copy of the live journal's rows from 2026-10-07 on — the same window that comment
    names — that volume is 586,937 B across 981 rows (mean 598.3 B), of which 326 rows
    sit exactly at the 200-char cap and every one of those is a `Bash` row. Worst case,
    if every at-cap row also carried a full field, the same window is 984,657 B at a
    1200-char ceiling (93.9 % of the bound) and 1,897,457 B at 4000 (181.0 %) — so 4000
    crosses the ceiling the fold exists to stay under, and 1200 does not. 1200 also
    covers the largest flag offset ever seen in a refused command on this box (1020,
    #3874 triage), which is where the number comes from.

    Asserting the constant is not decoration: it is the number the store-26 comment gets
    re-based against, so a future widening has to fail here first.
    """
    assert DJ.FULL_CMD_CHARS == 1200, f"FULL_CMD_CHARS is {DJ.FULL_CMD_CHARS}"
    assert DJ.FULL_CMD_CHARS > DJ.EXCERPT_CHARS == 200, (
        "the whole-command field must be strictly wider than the cut it replaces")
    import inspect

    # The bound has to be applied by the code that WRITES the row, not just named by a
    # constant: `record` spreads `_full_command_field`, and that helper is what cuts.
    assert "_full_command_field" in inspect.getsource(DJ.record), (
        "record() no longer spreads `_full_command_field`, so nothing applies the bound")
    assert "FULL_CMD_CHARS" in inspect.getsource(DJ._full_command_field), (
        "`_full_command_field` no longer cuts at FULL_CMD_CHARS, so the bound is only in "
        "the constant")


def test_a_list_shaped_command_writes_no_field_and_keeps_todays_excerpt(journal):
    """The scope rule is the `command` KEY, and a `command` that is not a string is not
    a command this module ever read: `excerpt_of` skips a LIST-valued `command` (it only
    accepts `isinstance(value, str)`) and falls through to the folded JSON of the whole
    arguments dict, so a field that joined the list would be a second rendering of the
    same call with no live row to compare it to — `~/lloyd-data/safety/denials.jsonl`
    held 0 rows whose excerpt begins with that folded-JSON form across its 981 rows on
    2026-11-07.

    Pinned both ways: the row is unchanged from today's shape, AND the excerpt really is
    the JSON fallback, so this node cannot quietly keep passing if `excerpt_of` one day
    starts joining lists — at which point the field has to be taught the shape too.
    """
    parts = ["rm", "-rf"] + [f"/home/alansrobotlab/lloyd-data/part{i:04d}" for i in range(40)]
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, _denies_whatever_it_is_asked_about, fail_closed=True)
    _run(hooks.fire_pre_tool_use(session_id="20261103_000008_chat", tool_name="Bash",
                                 tool_input={"command": parts}))

    rows = _rows(journal)
    assert len(rows) == 1, f"{len(rows)} rows for one refusal"
    assert rows[0]["excerpt"] == DJ.excerpt_of("Bash", {"command": parts})[:DJ.EXCERPT_CHARS]
    assert rows[0]["excerpt"].startswith('{"command"'), (
        "excerpt_of stopped folding a list command to JSON, so the scope rule below is "
        "the wrong rule any more — teach `command_of` the shape in the same commit")
    assert len(rows[0]["excerpt"]) == DJ.EXCERPT_CHARS, (
        "that fallback excerpt is not cut at today's cap, so this row's shape is not "
        "today's either")
    assert sorted(rows[0]) == _TODAYS_KEYS, (
        f"a list-valued command wrote {sorted(rows[0])}: the field is scoped to a string "
        "`command` key, and inventing a rendering for another shape is scope creep")
