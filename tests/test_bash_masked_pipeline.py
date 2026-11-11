"""#4049: a masked pipeline failure reaches the Bash tool's reported result.

bash reports a pipeline's LAST component's status, so `pytest -q | tail -45` on a
red suite lands as a green tool result and the pipe usually eats the failing text
as well. The fix under test is detection, never a semantics change: a foreground
command with a top-level `|` is run with three lines appended — one assignment
command capturing `$PIPESTATUS` and `$?`, one marker line, `exit $__lloyd_rc` — and
when the overall status is 0 while the FIRST component exited non-zero and not
141/SIGPIPE, one loud line is appended after the output.

Two process boundaries, and a test sits across each:

* **Python → bash argv.** The wrapper is only worth composing if bash receives it
  unharmed, and an upstream pass between the tool's spawn and `/bin/sh -c` consumes
  every *braced* reference (#2953). `test_what_bash_actually_receives…` spies the
  argv and asserts the unbraced capture and the re-exit are in it;
  `test_the_wrapper_carries_no_braced_reference…` runs the repo's own empty-expansion
  measurement over the composed text.
* **bash → the model's result, and on to the counter.** The real spawn runs real
  pipelines (no fake process can produce a `$PIPESTATUS`), and
  `test_the_persisted_row_is_what_the_counter_reads…` feeds the tool's own output
  through `build_tool_result_entry` and then through
  `app.tool_quality.is_masked_pipeline` — the row shape the daily sweep parses.

Every clause of the item is pinned here except the dated row, which is clause 5 of
`tests/test_tool_quality_sweep.py`:

1. a masked failure gets one appended line naming the component status and its own
   reach, with the same limitation in `_bash`'s docstring;
2. the reported status never changes — a failing pipeline keeps its identical
   `[exit code: N]` and its error flag, and a clean `true | true` is byte-identical
   (this is what pins the SINGLE-assignment capture and `exit $__lloyd_rc`:
   `__ps=$PIPESTATUS; __rc=$?` as two statements clobbers a real rc of 1 into 0,
   measured 2026-11-11, which would turn every failing pipeline green);
3. a SIGPIPE'd producer fires nothing — 141 is whitelisted, on the three commands
   measured on this box;
4. a command with no top-level `|`, and a command whose text runs into a
   here-document, are byte-identical: the capture is composed on the pipe shape and
   never into heredoc or quoted text;
5. — (the tool-quality row).

The blind spot is asserted too, because a partial instrument that reads as total
coverage is this box's filed defect class: a mid-stage failure in `a | b | c` and an
earlier pipeline in a multi-command script fire nothing, and the emitted line says
so in its own words.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_mcp.builtin_bash import _bash  # noqa: E402
from app.harness import bash_pipeline  # noqa: E402

NOTE_PREFIX = bash_pipeline.MASKED_NOTE_PREFIX
MARKER_TAG = bash_pipeline.MARKER_TAG


def run(command: str) -> str:
    """The tool's own answer for `command`, from a real spawn in a real bash.

    Not a faked subprocess: the whole instrument rests on `$PIPESTATUS`, which only
    a shell that actually ran the pipeline can supply. Foreground, sandbox settings
    as configured, one command per call — the same path `call_tool` takes.
    """
    return asyncio.run(_bash({"command": command, "timeout": 30000}))


def masked_line(text: str) -> str:
    """The one appended line, asserted to be exactly one and only then returned."""
    hits = [line for line in text.splitlines() if line.startswith(NOTE_PREFIX)]
    assert len(hits) == 1, f"expected exactly one note line in {text!r}"
    return hits[0]


# ───────────── clause 1: the line, its status, and its stated reach ───────────
def test_a_masked_failure_appends_one_line_naming_the_component_status():
    """`false | true` — the item's own acceptance command — is no longer green.

    The command reports exit code 0 (bash gives a pipeline the LAST component's
    status, and `true` is the last component) while its first component exited 1.
    Before this change the tool returned `(no output)` and nothing else; now the
    output is followed by exactly one line that names status 1.
    """
    out = run("false | true")
    assert out.startswith("(no output)")
    line = masked_line(out)
    assert "exited 1" in line
    assert "reported exit code 0" in line
    assert "[exit code:" not in out, "the reported status must not change"


def test_the_appended_line_is_the_shared_module_s_own_text() -> None:
    """The tool emits `masked_note`, not a copy of its wording.

    `app/harness/bash_pipeline.py` is imported by the tool and by the counter; a note
    spelled twice is a rate that silently stops matching what it counts.
    """
    out = run("false | true")
    assert bash_pipeline.masked_note(1) in out


def test_the_line_states_its_own_reach_first_component_of_the_last_pipeline_only():
    """A reader of the line alone can tell a clean reading from a covered command.

    The clause is the item's: an instrument must print its own blind spot. So the
    emitted sentence names the scope it does NOT cover — mid-stage failures,
    earlier pipelines, and SIGPIPE — in the same breath as the failure it found.
    """
    line = masked_line(run("false | true"))
    assert "FIRST component" in line
    assert "LAST pipeline only" in line
    assert "mid-stage" in line
    assert "earlier pipeline" in line
    assert "SIGPIPE (141)" in line


def test_bash_s_own_docstring_carries_the_same_limitation():
    """The reach clause is in `_bash`'s docstring, not only in the emitted line.

    The docstring is what the next reader of the tool reads before changing it; a
    claim about coverage that lives only in a runtime string is how a partial
    instrument comes to be trusted as a total one.
    """
    doc = _bash.__doc__ or ""
    assert doc, "_bash must have a docstring to carry the claim"
    for phrase in ("FIRST component", "LAST pipeline", "SIGPIPE", "141",
                   "middle stage", "here-document", "additive"):
        assert phrase in doc, f"docstring is missing {phrase!r}"
    assert "is_error" in doc, "the docstring must say the error flag does not move"


# ───────────── clause 2: the reported status is not this change's to touch ────
def test_a_failing_pipeline_keeps_its_identical_exit_code_and_gains_no_line():
    """`false | false` — both components fail, the pipeline exits 1.

    This is the pin on the capture's shape. `PIPESTATUS` and `?` are both reset by
    ANY simple command, a bare assignment included, so the obvious two-statement
    capture (`__ps=$PIPESTATUS` then `__rc=$?`) reads `$?` as the first assignment's
    0 and hands the tool `exit 0`: a red pipeline reported green, with a note
    asserting the opposite. Measured on this box 2026-11-11. The single assignment
    command is what keeps the real status, and `exit $__lloyd_rc` is what re-exits
    with it.
    """
    out = run("false | false")
    assert out.endswith("[exit code: 1]")
    assert NOTE_PREFIX not in out


def test_a_nonzero_pipeline_of_a_real_status_reports_that_status_unchanged():
    """The `[exit code: N]` suffix is the command's own, for N other than 1.

    `exit 7 | cat` — the first component exits 7, the pipeline exits 0 so a note
    WOULD be due, and the note must be the only thing added: no suffix, because the
    status bash reported really is 0.
    """
    out = run("sh -c 'exit 7' | cat")
    assert NOTE_PREFIX in out
    assert "exited 7" in masked_line(out)
    assert "[exit code:" not in out


def test_a_clean_pipeline_is_byte_identical_to_the_command_s_own_output():
    """`true | true` and `printf | cat` return exactly what they returned before.

    Byte-identical, not merely note-free: the marker line is stripped ahead of the
    truncation check, so the three appended lines contribute nothing to the bytes,
    to the emptiness test, or to the truncation decision.
    """
    assert run("true | true") == "(no output)"
    assert run("printf 'a\\nb\\n' | cat") == "a\nb\n"


def test_the_error_flag_follows_the_command_and_not_the_note():
    """A masked failure stays success-coded; a real failure stays error-coded.

    `call_tool` is the MCP boundary the harness reads: it returns
    `text_result(text, is_error=True if _bash_failed.get() else None)`, and
    `_bash_failed` is set on a non-zero exit and a timeout — never by this note,
    which is appended after the flag's only writer has run. When `is_error` is None
    `text_result` sniffs the payload instead, and the masked payload begins `(no
    output)\n[masked pipeline failure:` — not a JSON error object, so the sniff
    cannot turn it into a failure either. This is the seam at which the instrument
    would stop being additive if it ever set the flag, and it is the boundary the
    harness actually crosses to decide `is_error`.
    """
    import agent_mcp.builtin_bash as bb

    masked = asyncio.run(bb.call_tool("Bash", {"command": "false | true",
                                               "timeout": 30000}))
    assert masked.is_error is False, "a note must not flip the error flag"
    assert NOTE_PREFIX in masked.content[0].text

    failed = asyncio.run(bb.call_tool("Bash", {"command": "false | false",
                                               "timeout": 30000}))
    assert failed.is_error is True
    assert failed.content[0].text.endswith("[exit code: 1]")
    assert NOTE_PREFIX not in failed.content[0].text


def test_the_persisted_row_is_what_the_counter_reads():
    """Tool output → a persisted `role="tool"` row → `app.tool_quality`'s predicate.

    The daily sweep parses rows another process wrote, so the seam that decides
    whether the rate exists at all is whether a row holding the tool's real answer
    is a row the counter recognises. The row is built in the on-disk shape — `content`
    plus a `stats` dict, as `app/transcript_entries.build_tool_result_entry` writes
    it, with the same keys a real transcript carries — and handed to
    `is_masked_pipeline` and `command_takes_a_pipe` exactly as the classifier hands
    them over one call.
    """
    from app import tool_quality

    masked = run("false | true")
    clean = run("printf 'a\\nb\\n' | cat")

    def row(call_id: str, command: str, text: str) -> tuple[dict, dict]:
        """One call/result pair in the shape a session file really carries.

        A call row's tool call is `{"id", "call_id", "type": "function", "function":
        {"name", "arguments"}}` with `arguments` the JSON STRING the model produced —
        the same object `is_degenerate_arguments` parses, and the reason
        `_command_argument` reads a string and not a dict. A result row is `content`
        plus the `stats` dict `build_tool_result_entry` writes.
        """
        return ({"id": f"msg_{call_id}_tc", "role": "assistant",
                 "content": [{"type": "text", "text": ""}],
                 "tool_calls": [{"id": call_id, "call_id": call_id,
                                 "type": "function",
                                 "function": {"name": "Bash",
                                              "arguments": json.dumps(
                                                  {"command": command})}}],
                 "timestamp": "2026-11-11T00:00:00", "stats": {"iteration": 1}},
                {"id": f"msg_{call_id}_tr", "role": "tool",
                 "tool_call_id": call_id, "name": "Bash", "content": text,
                 "stats": {"result_chars": len(text), "raw_chars": len(text),
                           "is_error": False}})

    call_m, result_m = row("c1", "false | true", masked)
    call_c, result_c = row("c2", "printf 'a\\nb\\n' | cat", clean)
    # `classify_transcript` hands the predicates the individual tool_call object, not
    # the assistant message that carries it, so that is the unit asserted here.
    assert tool_quality.is_masked_pipeline(result_m) is True
    assert tool_quality.is_masked_pipeline(result_c) is False
    assert tool_quality.command_takes_a_pipe(call_m["tool_calls"][0]) is True
    assert tool_quality.command_takes_a_pipe(call_c["tool_calls"][0]) is True

    rows = tool_quality.classify_transcript(
        {"session_id": "s", "messages": [call_m, result_m, call_c, result_c]})
    assert rows["Bash"]["with_pipe"] == 2
    assert rows["Bash"]["masked"] == 1, "one real masked failure, counted once"


# ────────────── clause 3: a SIGPIPE'd producer is not a masked failure ────────
@pytest.mark.parametrize("command", [
    "seq 1 200000 | head -1",
    "yes 2>/dev/null | head -3",
    "ls -la /etc | head -2",
])
def test_a_sigkilled_by_sigpipe_producer_fires_nothing(command: str):
    """The three commands measured on this box at ps=141: benign by design.

    `cmd | head -1` kills the writer with SIGPIPE and bash records 141 for it — the
    normal shape of half of this tool's traffic, not a failure. Without the
    whitelist the note would fire on most of them, and an instrument nobody trusts
    is worse than the silent one it replaced.
    """
    out = run(command)
    assert NOTE_PREFIX not in out
    assert MARKER_TAG not in out
    assert "[exit code:" not in out


def test_the_whitelist_is_the_number_sigpipe_and_no_other():
    """141 is exempted because it is 128+SIGPIPE, and nothing else is exempted.

    The predicate is stated once, in the module, and this pins both edges: 141
    returns no note, and 140 (128+SIGUSR1-ish neighbour, and any other non-zero)
    does. A whitelist that widened silently is how the instrument would go mute.
    """
    assert bash_pipeline.SIGPIPE_STATUS == 141
    assert bash_pipeline.masked_note_for(141, 0) is None
    assert bash_pipeline.masked_note_for(0, 0) is None
    assert bash_pipeline.masked_note_for(140, 0) is not None
    assert bash_pipeline.masked_note_for(1, 0) is not None
    # A non-zero overall status is the command's own story, note or no note.
    assert bash_pipeline.masked_note_for(1, 1) is None


# ──── clause 4: no top-level pipe, no wrapper — and never into a heredoc ─────
def test_a_command_with_no_top_level_pipe_is_byte_identical():
    """`echo` and `||`: no capture is composed, so the bytes cannot change.

    `false || true` is the sharp half of this: it contains a `|` character, and a
    scanner that read `||` as a pipe would wrap commands that never piped anything.
    Element 0 of that construct is not a producer, and the wrapper would report a
    status that means something else entirely.
    """
    assert run("echo hello-pipe-check") == "hello-pipe-check\n"
    assert run("false || true") == "(no output)"
    assert run("echo 'quoted | pipe' | cat") == "quoted | pipe\n"


def test_a_command_whose_text_runs_into_a_heredoc_is_byte_identical():
    """A here-document absorbs anything appended to the command: no wrapper at all.

    Two shapes. A document that runs to the end of the text would swallow the
    capture as heredoc CONTENT, and `cat <<E | wc -l` has a `|` that belongs to no
    pipeline whose first component bash could report. Both must return today's
    bytes: the count, and the two echoed lines, with no marker and no note.
    """
    assert run("cat <<'E' | wc -l\none\ntwo\nE") == "2\n"
    assert run("echo one\ncat <<'E'\ninner\nE") == "one\ninner\n"


def test_the_shape_check_refuses_the_shapes_appending_would_change():
    """The pure predicate behind clause 4, including the dangling-operator arm.

    `cmd \\` would continue onto the capture and run it as an environment prefix; a
    trailing `|` or `&&` would swallow it as an operand. Those commands are not
    pipelines in the sense this instrument can speak about, and appending to them
    would change what parses rather than observe it.
    """
    assert bash_pipeline.has_top_level_pipe("false | true") is True
    assert bash_pipeline.has_top_level_pipe("cmd |& grep x") is True
    assert bash_pipeline.has_top_level_pipe("false || true") is False
    # `echo a|b` IS a pipeline to bash — `echo a` fed to a command named `b` — so the
    # scan composes onto it. A wrong "yes" costs a wrapper that never fires; a wrong
    # "no" costs a masked failure nobody ever sees, which is the asymmetry the scan is
    # written to favour. The genuinely quoted and bracketed forms are the ones refused.
    assert bash_pipeline.has_top_level_pipe("echo a|b") is True
    assert bash_pipeline.has_top_level_pipe("echo ${x//a|b}") is False
    assert bash_pipeline.has_top_level_pipe("cat <<E | wc -l\nx\nE") is False
    assert bash_pipeline.has_top_level_pipe("echo 'unbalanced | ") is False
    assert bash_pipeline.has_top_level_pipe("false |") is False
    assert bash_pipeline.has_top_level_pipe("false &&") is False
    assert bash_pipeline.has_top_level_pipe("make \\") is False
    assert bash_pipeline.has_top_level_pipe("") is False
    assert bash_pipeline.capture_wrapper("echo hi") is None


# ─────────────── the blind spot, asserted rather than implied ─────────────────
def test_a_mid_stage_failure_fires_nothing_and_that_is_documented():
    """`true | false | true`: the middle stage failed and element 0 did not.

    Only `$PIPESTATUS` element 0 is reachable — the braced array spellings are
    consumed by the env-substitution pass, measured again on this box 2026-11-11 —
    so a mid-stage failure is invisible. This test asserts the instrument does NOT
    claim it, which is what makes the emitted reach clause true rather than
    decorative.
    """
    out = run("true | false | true")
    assert NOTE_PREFIX not in out


def test_an_earlier_pipeline_in_a_multi_command_script_fires_nothing():
    """The item's own acceptance case: the LAST pipeline is clean, an earlier one was not.

    `false | true; echo done` — the masked failure is real, and `PIPESTATUS` by the
    time the capture runs describes `echo done`, so the tool stays silent. The
    limitation is in the emitted line, in `_bash`'s docstring and here; widening past
    the last pipeline is the follow-up #4049 step 5 defers until a measured rate
    exists.
    """
    out = run("false | true; echo done")
    assert out == "done\n"
    assert NOTE_PREFIX not in out


# ─────────── the boundary: what bash is actually handed, and why ─────────────
def _handed_command(argv: "list[str]") -> str:
    """The shell text inside a spawn argv: whatever follows the last `-c`.

    The plain path spawns through `_path_sandbox.wrap`, which prefixes bwrap flags
    (`bwrap … -- /bin/sh -c <text>`), and the sandboxed path through
    `_tool_sandbox.bwrap_argv`, which prefixes its own binds. Both end in a `-c` and
    one argument, so that last element is the text the shell parses on either path.
    """
    return argv[len(argv) - argv[::-1].index("-c")]


def _spy_spawns(monkey: "pytest.MonkeyPatch", *, sandboxed: bool = False):
    """Install an argv spy on the real spawn, and return the collected argvs.

    The spy delegates to the real `create_subprocess_exec`, so the command still
    runs in a real bash and the `$PIPESTATUS` the wrapper reads is the real one; only
    the argv is recorded on its way past.
    """
    import asyncio as aio
    import contextvars

    import agent_mcp.builtin_bash as bb

    seen: "list[list[str]]" = []
    real_exec = aio.create_subprocess_exec

    async def spy(*argv, **kwargs):
        seen.append(list(argv))
        return await real_exec(*argv, **kwargs)

    monkey.setattr(aio, "create_subprocess_exec", spy)
    if sandboxed:
        # The sandboxed arm needs bwrap to exist as far as the tool is concerned, and
        # `bwrap_argv` replaced by an argv that really runs the command: what is under
        # test is the STRING handed to that builder, not bwrap itself.
        monkey.setattr(bb._tool_sandbox, "current_sandboxed",
                       contextvars.ContextVar("sb", default=True))
        monkey.setattr(bb._tool_sandbox, "sandbox_available",
                       lambda: (True, ""), raising=False)
        monkey.setattr(bb._tool_sandbox, "bwrap_argv",
                       lambda command, cwd: ["/bin/sh", "-c", command])
        monkey.setattr(bb._resource_ceiling, "apply", lambda argv: (argv, None))
    return seen


def test_what_bash_actually_receives_is_the_command_plus_the_capture():
    """The wrapper survives the trip from Python to the shell, on the plain path.

    The text handed to the spawn must be the caller's command followed by the
    single-assignment capture, the marker printf and the re-exit — and the result
    must still carry the note, which is the proof that the same text is what bash
    parsed rather than something reconstructed afterwards.
    """
    monkey = pytest.MonkeyPatch()
    try:
        seen = _spy_spawns(monkey)
        out = asyncio.run(_bash({"command": "false | true", "timeout": 30000}))
    finally:
        monkey.undo()

    assert seen, "the tool must spawn a shell"
    handed = _handed_command(seen[0])
    assert handed.startswith("false | true")
    assert "__lloyd_ps=$PIPESTATUS __lloyd_rc=$?" in handed
    assert "exit $__lloyd_rc" in handed
    assert MARKER_TAG in handed
    assert NOTE_PREFIX in out


def test_the_sandboxed_path_is_handed_the_same_wrapped_text():
    """The bwrap arm is handed the wrapped command too, and still annotates.

    Both spawn paths build their argv differently (`bwrap_argv` versus
    `_path_sandbox.wrap`), and both are given `exec_command`. If only one received
    the capture, whether a masked failure surfaced would depend on whether the box
    runs bwrap — a difference no caller could see coming, and the reason the wrapper
    is composed once, above the branch.
    """
    monkey = pytest.MonkeyPatch()
    try:
        seen = _spy_spawns(monkey, sandboxed=True)
        out = asyncio.run(_bash({"command": "false | true", "timeout": 30000}))
    finally:
        monkey.undo()

    assert seen, "the sandboxed path must still spawn"
    handed = _handed_command(seen[0])
    assert "__lloyd_ps=$PIPESTATUS __lloyd_rc=$?" in handed
    assert "exit $__lloyd_rc" in handed
    assert NOTE_PREFIX in out


def test_a_command_with_no_pipe_is_never_wrapped_at_the_spawn():
    """The cheapest half of the byte-identical guarantee, checked at the argv.

    The wrapper is composed on a shape check precisely so that the overwhelming
    majority of Bash calls — no pipeline in them — are handed to bash exactly as
    written. A capture appended to everything would be a change to every command's
    output, which is the opposite of this item.
    """
    monkey = pytest.MonkeyPatch()
    try:
        seen = _spy_spawns(monkey)
        asyncio.run(_bash({"command": "echo no-pipe-here", "timeout": 30000}))
    finally:
        monkey.undo()

    assert _handed_command(seen[0]) == "echo no-pipe-here"


def test_the_wrapper_carries_no_braced_reference_the_prepass_could_consume():
    """The composed text passes the repo's own empty-expansion measurement.

    The pass between the tool's spawn and `/bin/sh -c` consumes braced references, so
    `${PIPESTATUS[@]}` and `${PIPESTATUS[*]}` arrive empty and the whole instrument
    would read "nothing failed" forever. What survives is bare `$PIPESTATUS` — bash's
    own shorthand for element 0 — and `check_empty_expansion_measurement`, the
    in-repo measurement of that very grammar, is run here over the wrapper.
    """
    from app.harness.env_expansion import check_empty_expansion_measurement

    wrapper = bash_pipeline.capture_wrapper("pytest -q 2>&1 | tail -45")
    assert wrapper is not None
    assert "${" not in wrapper, "a braced reference would reach bash empty"
    assert "$PIPESTATUS" in wrapper
    assert "$?" in wrapper
    assert check_empty_expansion_measurement(wrapper) is None
