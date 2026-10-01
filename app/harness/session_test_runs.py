"""Which test runs did THIS session execute, and did each one pass?

Item #1950. The parent #1947 wanted a policy gate that refuses an unattended
`git push` unless the session's own record holds a passing test run. The store
it would have read — `agent_mcp/_tool_effects.py`, the exactly-once effect
ledger — cannot hold that precondition at all: it records only tools the
annotations table classifies `side_effecting`, and `side_effecting("Bash")` is
False (measured: `Edit` and `Write` too; `email_send` and `backlog_write_task`
True). The live table agrees — every row it holds belongs to an MCP tool and
none of them is `Bash`. Marking `Bash` side-effecting would not fix that either
and must not be done: the ledger is an exactly-once *effect* guard, so a Bash
row would make a repeated identical command inside one effect scope suppress or
replay, which is to say re-running `pytest` in a turn would be refused. This
module therefore sits BESIDE the ledger and reads a substrate that can hold a
Bash call at all: the session document's own `assistant.tool_calls` /
`role: "tool"` events, the same events `app/harness/action_review.py` taps.

Two failure modes this file exists to refuse:

**The exit code is not the verdict.** `agent_mcp/builtin_bash.py` sets
`is_error` from the shell's return code, and a pipeline returns the LAST
command's status, so a suite piped through `head` or `tail` exits 0 however red
it was. This is not a hypothetical: measured over `~/lloyd-data/sessions` on
2026-10-01 (4,056 documents, 98,368 tool calls), **1,758** stored results from a
test-shaped Bash call carry `stats.is_error: false` while their own text reports
a failure count (`2 failed`, `8 failed`, …) — 1,758 runs that a reader trusting
the flag would file as green suites. So `is_error` is treated as
necessary-but-not-sufficient: `is_error=True` is never passing, whatever the
text claims, and a text failure summary overrides a clean exit code.

**Absence is not a pass, and a truncated result is not a summary.** A session
that ran no test-shaped command has ZERO runs, and the answer to "did a test
run pass here" for it is False — not "no failing run found, so yes". The same
rule is applied to a result whose text was spilled to disk rather than stored:
`app/harness/tool_result_spill.py` keeps `PREVIEW_CHARS = 2_000` of head and
moves the rest to a sidecar (and `app/harness/microcompact.py` writes the same
`Full output saved to:` pointer for older turns), while the pytest summary line
lives at the END of the output — so a spilled red suite would otherwise look
clean. Such a run is reported UNDETERMINED, which is neither passing nor failing,
and it counts toward the denominator.

No gate consumes this yet, deliberately: the corpus the item was measured over
holds no executed `git push`, so wiring one is a ruling for after this has been
read back against live transcripts.

Run it over one session::

    ~/lloyd/.venvs/lloyd/bin/python -m app.harness.session_test_runs <session_id>

The CLI prints the directory and the tree it read, and exits 3 when no document
is there, because `app.paths` anchors `SESSIONS_DIR` to whichever checkout is
importing it: run from a linked worktree and the default root is that worktree's
own `.lloyd-data/sessions`, usually empty, where "0 test run(s)" would otherwise
be indistinguishable from a session that never tested — the same false zero the
reader exists to refuse, arriving by way of the working directory.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# A plain import, not a guarded one. `app/paths.py` imports only pathlib and os,
# so there is nothing to guard against; and the fallback it used to carry —
# `Path.home() / "lloyd-data" / "sessions"` — was the worse failure, because HOME
# is redirected in a gate round and in any sandboxed run, so a reader that could
# not import paths would silently measure someone else's transcript directory
# instead of saying it had none. `describe_tree` is printed by the CLI for the
# reason in the docstring: SESSIONS_DIR resolves inside whichever checkout is
# importing this module.
from app.paths import SESSIONS_DIR, describe_tree

#: A Bash command that runs a test suite. Shaped to catch `python -m pytest`,
#: a bare `pytest`, `pytest -k …`, `go test ./...`, `npm test` and `make test`,
#: and deliberately NOT the word `test` inside a path or a `grep test` — the
#: reader's whole value is that a run it counts was a run that could fail.
TEST_CMD_RX = re.compile(
    r"(?:^|[\s;&|(])(?:python\d?(?:\.\d+)*\s+-m\s+)?pytest\b"
    r"|(?:^|[\s;&|(])pytest\b"
    r"|(?:^|[\s;&|(])go\s+test\b"
    r"|(?:^|[\s;&|(])npm\s+(?:run\s+)?test\b"
    r"|(?:^|[\s;&|(])make\s+(?:test\b|pytest\b)"
)

#: A result text that reports tests. Anchored on pytest's own vocabulary: a
#: count line, a collection error, or a traceback header.
TEST_TEXT_RX = re.compile(
    r"\b\d+ (?:passed|failed|error|errors|skipped|deselected|xfail)"
    r"|\b(no tests ran|collection error)\b"
    r"|^=+ ERRORS =+$|Traceback \(most recent call last\)",
    re.M,
)

#: Failure evidence in the text. `N failed`, `N errors`, a `FAILED node` line,
#: an `ERRORS` banner. `failed` is matched with a count so a prose sentence
#: mentioning a failed test inside passing output does not read as a red suite.
FAIL_TEXT_RX = re.compile(
    r"\b\d+ (?:failed|errors?)\b"
    r"|^(?:FAILED|ERROR) \S+"
    r"|^=+ (?:FAILURES|ERRORS) =+$",
    re.M,
)

#: Success evidence: pytest's summary line, e.g. `8 passed in 1.2s`.
PASS_TEXT_RX = re.compile(r"^=+ \d+ passed.* in [\d.]+s =+$", re.M)

#: The two ways this tree stores a result it chose not to keep whole, each named
#: by the function that writes it. `app/harness/tool_result_spill.py::maybe_spill`
#: emits a `<persisted-output>` tag, an `Output too large (…, … chars).` line, a
#: `Full output saved to: <path>` pointer and a `Preview (first 2.0 KB):` block —
#: three of the four alternatives below are that one stub's own text. And
#: `app/transcript_entries.py::truncate_tool_result` performs the hard tail cut,
#: keeping `TOOL_RESULT_MAX_CHARS = 2000` and appending `...(truncated)`, which is
#: what a long result gets today when spill is off (`transcript_spill_enabled`).
#: A result carrying either marker is not a short result that passed — it is a
#: long one we cannot read, and the summary line that would settle it sat at the
#: end, which is precisely what was cut.
TRUNCATED_RX = re.compile(
    r"<persisted-output>|Output too large|Full output saved to:"
    r"|\.\.\.\(truncated\)"
)


def _text_of(content: Any) -> str:
    """Flatten a stored tool result to plain text.

    `content` is either a string or the block list the model sees,
    `[{"type": "text", "text": ...}, …]`; both shapes occur in the corpus.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def _is_error_of(message: dict) -> bool:
    """The recorded error flag, wherever this tree writes it.

    Read from `stats.is_error`, which is where the session writer puts it on a
    `role: "tool"` message; a top-level `is_error` is also honoured, since the
    in-memory event carries that name and a fixture may too.
    """
    stats = message.get("stats")
    if isinstance(stats, dict) and "is_error" in stats:
        return bool(stats.get("is_error"))
    return bool(message.get("is_error"))


def classify_test_run(*, content: Any, is_error: bool) -> str:
    """`passing` / `not-passing` / `undetermined` for one test result.

    Order matters, and each step is a refusal of a specific wrong answer:

    1. `is_error` → not-passing. The tool said it failed; no text can un-fail it.
    2. failure evidence in the text → not-passing, EVEN THOUGH `is_error` is
       false. This is the pipe-masked case, and the one that would otherwise
       file a red suite as a satisfied precondition.
    3. text truncated before its end → undetermined. The summary line we did not
       see may have said `2 failed`; guessing "passing" from "we saw no failure"
       is the zero-denominator error in a different costume.
    4. an explicit pytest summary naming no failure → passing.
    5. clean exit and no recognisable summary → undetermined. Not a pass: we
       cannot tell a green suite from a suite that never ran.
    """
    text = _text_of(content)
    if is_error:
        return "not-passing"
    if FAIL_TEXT_RX.search(text):
        return "not-passing"
    if TRUNCATED_RX.search(text):
        return "undetermined"
    if PASS_TEXT_RX.search(text) and not re.search(r"\b\d+ failed\b", text):
        return "passing"
    if re.search(r"^\d+ passed\b", text, re.M):
        return "passing"
    return "undetermined"


@dataclass(frozen=True)
class TestRun:
    """One test run a session executed, and what we can say about it."""

    call_id: str
    command: str
    is_error: bool
    verdict: str
    truncated: bool
    turn_id: str = ""

    @property
    def passed(self) -> bool:
        """True only when the evidence says the whole suite was green.

        `undetermined` is False here on purpose: the consumer's question is
        "may I rely on a green suite", and an unread result is not one. The
        distinction is kept in `verdict` so a caller that wants to distinguish
        "red" from "unverifiable" can.
        """
        return self.verdict == "passing"


def _commands_by_call(messages: Iterable[dict]) -> dict[str, tuple[str, str, str]]:
    """`call_id -> (tool_name, command, turn_id)` for every tool call."""
    out: dict[str, tuple[str, str, str]] = {}
    for m in messages:
        for call in (m.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            cid = str(call.get("id") or "")
            if not cid:
                continue
            raw = fn.get("arguments")
            if isinstance(raw, str):
                # On disk this is a JSON document nested inside the transcript's
                # own JSON string, which is why its quotes arrive escaped — the
                # false zero recorded in
                # `knowledge/software/session-tool-traffic-counting.md`: a census
                # grepping `"command": "` matches nothing corpus-wide.
                # Measured over `~/lloyd-data/sessions` on 2026-10-01 (4,056
                # documents): `function.arguments` is a str in 98,368 of 98,368
                # tool calls and all 98,368 parse as JSON, so the `except` below
                # guards a truncated or hand-built document, not a shape seen in
                # the wild.
                try:
                    args = json.loads(raw)
                except ValueError:
                    args = {}
            elif isinstance(raw, dict):
                # Not the on-disk shape — an in-process caller's shape. Kept
                # because a caller that already holds parsed arguments should
                # not have to serialise them to be read.
                args = raw
            else:
                args = {}
            command = str(args.get("command") or "")
            out[cid] = (str(fn.get("name") or ""), command,
                        str(m.get("turn_id") or ""))
    return out


def session_test_runs(session_id: str, *,
                      sessions_dir: Path | None = None) -> list[TestRun]:
    """Every test run in one session document, oldest first.

    Returns an EMPTY list for a session that ran nothing test-shaped, and for a
    session file that is absent — both are reported as zero runs by `summarise`,
    never as a satisfied precondition.
    """
    root = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
    path = root / f"{session_id}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    messages = data.get("messages") or []
    commands = _commands_by_call(messages)

    runs: list[TestRun] = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        cid = str(m.get("tool_call_id") or "")
        name, command, turn = commands.get(cid, ("", "", ""))
        if name != "Bash" or not command:
            continue
        if not TEST_CMD_RX.search(command):
            continue
        content = m.get("content")
        is_error = _is_error_of(m)
        text = _text_of(content)
        runs.append(TestRun(
            call_id=cid,
            command=command,
            is_error=is_error,
            verdict=classify_test_run(content=text, is_error=is_error),
            truncated=bool(TRUNCATED_RX.search(text)),
            turn_id=turn))
    return runs


def summarise(session_id: str, *,
              sessions_dir: Path | None = None) -> dict[str, Any]:
    """The reader's answer, with its denominator beside every figure.

    `has_passing_run` is the only field a future gate should reach for, and it
    is False for a session with no runs: the question was "did a test run
    pass", not "did anything fail to fail".
    """
    runs = session_test_runs(session_id, sessions_dir=sessions_dir)
    passing = [r for r in runs if r.verdict == "passing"]
    failing = [r for r in runs if r.verdict == "not-passing"]
    undetermined = [r for r in runs if r.verdict == "undetermined"]
    return {
        "session_id": session_id,
        "runs": len(runs),
        "passing": len(passing),
        "not_passing": len(failing),
        "undetermined": len(undetermined),
        "has_passing_run": bool(passing),
        "last_run_verdict": runs[-1].verdict if runs else "no-runs",
        # Every rate this reader could ever print divides by `runs`; a caller
        # that wants a ratio must check that number first.
    }


def _cli(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    sid = argv[1]
    root = Path(argv[2]) if len(argv) > 2 else SESSIONS_DIR
    path = root / f"{sid}.json"
    print(f"reading {path}  ({describe_tree()})")
    if not path.is_file():
        # `0 runs` from an absent document is the false zero, not the answer:
        # name the tree so the reader can tell "not here" from "ran nothing",
        # and leave a non-zero status so no script can take it as a clean run.
        print(f"NO SESSION DOCUMENT at {path} — nothing is reported, because "
              "an absent file is not a session that ran no tests")
        return 3
    runs = session_test_runs(sid, sessions_dir=root)
    s = summarise(sid, sessions_dir=root)
    print(f"session {sid}: {s['runs']} test run(s) "
          f"[passing {s['passing']} | not-passing {s['not_passing']} "
          f"| undetermined {s['undetermined']}]")
    print(f"has_passing_run: {s['has_passing_run']}  "
          f"(denominator: {s['runs']} runs)")
    for r in runs:
        flag = "TRUNCATED" if r.truncated else "full"
        print(f"  {r.verdict:12s} is_error={str(r.is_error):5s} {flag:9s} "
              f"{r.command[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
