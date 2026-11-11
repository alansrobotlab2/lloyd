"""The Bash tool's masked-pipeline instrument: one pure string layer (#4049).

A pipeline's exit status is its **last** component's status unless `pipefail` is
set, so `pytest -q | tail -45` on a red suite is reported by the shell as success
and the tool's `[exit code: N]` suffix never fires. The pipe usually eats the
error text as well, so the model reads three harmless lines and says the suite is
green. This box has already been cut by exactly that (`if pytest "$NODE" -q 2>&1 |
tail -45; then echo "green alone"` labelled two red nodes green-alone on
2026-11-01), and answered it with a rule for the model rather than a fix for the
tool — which is what this module replaces.

**Detection, never a semantics change.** Nothing here touches the reported exit
status: the composed wrapper re-exits with the status the command already had, so
`rc`, the `[exit code: N]` suffix and the `is_error` flag are what they were
before this file existed. What is added is one line after the output when the
overall status is 0 while the *first* component of the *last* pipeline exited
non-zero for a reason other than SIGPIPE. Blanket `set -o pipefail` is deliberately
NOT used: it changes the status of commands agents and grader scripts already
branch on, including their own `if` conditions — the behaviour change
`tests/test_flash_next_launcher.py` exists to avoid.

**This layer is shared by two readers, which is why it is a module and not a
closure inside `agent_mcp.builtin_bash`.** The tool composes the wrapper and
strips its marker; `app/tool_quality.py` re-reads the same shape out of a
transcript (`has_top_level_pipe` over the recorded command,
`MASKED_NOTE_PREFIX` over the recorded result) to count the rate. A second
implementation of either test is a rate that silently stops matching the thing
being counted.

**The marker, and why it rides in the output.** The tool needs the component
status back across the spawn, and the only channel that already exists is stdout.
The wrapper therefore prints one line, `__LLOYD_PIPE__ <component> <overall>`,
which the tool strips off the tail before the truncation check — so the bytes the
model sees, and the truncation decision itself, are the ones the command wrote.
The marker carries the overall status as well as the component status: the tool
uses `proc.returncode` for every verdict it makes, and a marker whose overall
status disagrees with the spawn is discarded rather than trusted, because a
wrapper that cannot reproduce the real status has no business commenting on it.

**Spelling is measured, not assumed.** A substitution pass sits between the tool's
spawn and `/bin/sh -c` and consumes every *braced* reference (#2953,
`scripts/bash_tool_prepass_probe.py`), so `${PIPESTATUS[@]}` and `${PIPESTATUS[*]}`
arrive empty while bare `$PIPESTATUS` reaches bash intact and reads element 0 —
measured 2026-11-11 on this box, and re-pinned by
`tests/test_bash_masked_pipeline.py`. `/bin/sh -> bash` here, which is the only
reason bash's array works at all. Both captures are therefore unbraced, and they
are taken in ONE assignment command: `PIPESTATUS` and `?` are reset by any simple
command, a bare assignment included, so the two-statement form
`__ps=$PIPESTATUS; __rc=$?` reports 0 for a pipeline that exited 1 — measured.
One command is correct because both right-hand sides are expanded before either
name is bound.

**Reach, stated wherever the instrument speaks.** `PIPESTATUS` describes the last
pipeline executed, and only element 0 is readable here, so a mid-stage failure in
`a | b | c`, an earlier pipeline in a multi-command script, a command with no
top-level `|`, and anything the shape check refuses (a heredoc anywhere in the
text, an unbalanced quote, a dangling operator) are all invisible. A partial
instrument prints its own blind spot; an instrument that reads as total coverage
is this box's filed defect class.
"""
from __future__ import annotations

import re

#: A producer killed by SIGPIPE reports 128+13. `seq 1000 | head -1` leaves the
#: writer in that state BY DESIGN, so without this exemption half of the Bash
#: tool's traffic (`| head`, `| head -1`, `| grep -m1`) would arrive with a
#: warning, and a noisy instrument is an instrument everybody ignores.
SIGPIPE_STATUS = 141

#: The marker's own tag. Deliberately unlikely in command output and fixed, so
#: `tests/test_bash_masked_pipeline.py` can pin both the emitter and the parser.
MARKER_TAG = "__LLOYD_PIPE__"

#: The three lines appended to a command that passed the shape check. One
#: assignment command for both captures (see the module docstring), then the
#: marker, then the re-exit with the status the command already had.
_WRAPPER = (
    "__lloyd_ps=$PIPESTATUS __lloyd_rc=$?\n"
    f"printf '\\n{MARKER_TAG} %s %s\\n' \"$__lloyd_ps\" \"$__lloyd_rc\"\n"
    "exit $__lloyd_rc"
)

_MARKER_RE = re.compile(
    r"\n" + re.escape(MARKER_TAG) + r" (?P<ps>\d{1,3}) (?P<rc>\d{1,3})\n\Z")

#: What the appended line opens with. `app/tool_quality.py` counts occurrences of
#: this exact string in a persisted Bash result, so the note and the counter share
#: one definition and one edit cannot desync them.
MASKED_NOTE_PREFIX = "[masked pipeline failure:"

#: A `|` inside `${…}` is a parameter transformation (`${x//a|b}`), not a pipe.
_SIGIL_BRACE = re.compile(r"\$\{")
_SIGIL_SUB = re.compile(r"\$\(")

#: Trailing spellings that leave the command mid-expression. Appending a line to
#: one of these would not observe a masked failure, it would change what parses:
#: `cmd \` continues onto the capture and runs it as an environment prefix, and a
#: dangling `|` or `&&` swallows the capture as its right-hand operand.
_DANGLING_TAIL = ("|", "&", "&&", "||", ";", "<", ">", "\\")


def _scan(command: str) -> "tuple[bool, bool, bool]":
    """One left-to-right pass. Returns `(pipe, heredoc, unbalanced_quote)`.

    `pipe` is True when a `|` was seen at shell level: outside single and double
    quotes, outside `$( )` and `${ }`, and not as part of `||`. A `|&` (bash's
    stderr-pipe) counts as a pipeline, which is what it is.

    Deliberately not a shell parser: `case` patterns and here-strings inside an
    arithmetic expression can raise a false `pipe`, and a false `pipe` costs only
    a wrapper that never fires (the note still requires a non-zero component
    status), while a false *absence* costs a masked failure nobody ever sees. So
    the scan errs towards composing.
    """
    depth = 0
    in_single = False
    in_double = False
    saw_pipe = False
    saw_heredoc = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_single:
            i += 2
            continue
        if in_single:
            if ch == "'":
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = not in_double
            i += 1
            continue
        if in_double:
            # Inside double quotes `$( … )` still runs a shell, and a `|` there is
            # a real pipe inside a subshell whose status is NOT element 0 of the
            # outer pipeline — so it must not read as a top-level pipe.
            if _SIGIL_SUB.match(command, i):
                depth += 1
                i += 2
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            i += 1
            continue
        if command.startswith("<<", i):
            if not command.startswith("<<<", i):
                # A real here-document: its body runs to a terminator line, so any
                # appended text becomes heredoc CONTENT rather than a command.
                return False, True, False
            i += 3
            continue
        if _SIGIL_SUB.match(command, i):
            depth += 1
            i += 2
            continue
        if _SIGIL_BRACE.match(command, i):
            depth += 1
            i += 2
            continue
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            i += 1
            continue
        if ch == "|" and depth == 0:
            if command.startswith("||", i):
                # An OR list, not a pipeline: `false || true` has no producer whose
                # status element 0 could name, so its second `|` must not be read as
                # a pipe on the next iteration either.
                i += 2
                continue
            saw_pipe = True
        i += 1
    return saw_pipe, saw_heredoc, bool(in_single or in_double)


def has_top_level_pipe(command: str) -> bool:
    """True when the capture is safe and worth composing onto `command`.

    The cheap shape check the whole instrument hangs on: it is what keeps a
    command with no pipeline byte-identical, since the wrapper is appended to
    nothing unless this returns True. False for a heredoc anywhere in the text
    (the capture would land inside the document, and a document that runs to EOF
    would eat it: `cat <<E | wc -l` is rare next to `grep x f | head`, and the
    heredoc's own `|` is not a pipe in the first place), for an unbalanced quote,
    and for a command left mid-expression.
    """
    if not command:
        return False
    pipe, heredoc, unbalanced = _scan(command)
    if not pipe or heredoc or unbalanced:
        return False
    tail = command.rstrip()
    return not any(tail.endswith(op) for op in _DANGLING_TAIL)


def capture_wrapper(command: str) -> "str | None":
    """`command` with the status capture appended, or None when it must not be.

    `command` itself is never returned with a trailing newline stripped: a command
    ending in a line continuation stays a line continuation, and the blank line
    this appends after a newline-terminated command is worth one discarded parse.
    """
    if not has_top_level_pipe(command):
        return None
    return command + "\n" + _WRAPPER


def split_marker(output: str) -> "tuple[str, int | None, int | None]":
    """Peel a trailing marker off `output`.

    Returns `(rest, component_status, overall_status)`, with `(output, None, None)`
    when no marker is there — a command that wrote to the pty after the marker, a
    killed shell that never reached the printf, or a wrapper that was never
    composed all look like this, and the answer in every case is today's output
    unchanged. The peeled bytes include the newline the marker's own `printf`
    contributed, so `rest` is exactly what the command wrote: truncation,
    emptiness and the `(no output)` substitution all see those bytes.
    """
    match = _MARKER_RE.search(output)
    if match is None:
        return output, None, None
    return (output[:match.start()], int(match.group("ps")),
            int(match.group("rc")))


def masked_note_for(component_status: "int | None", rc: int) -> "str | None":
    """The line to append for one finished command, or None.

    Only three facts decide it, and each is a clause of #4049: an overall status
    of 0 (a failing command already gets `[exit code: N]`, and this line must not
    ride along with it — the additive rule and the byte-identical clause both
    depend on that), a component status that is not 0, and not 141.
    """
    if rc != 0:
        return None
    if component_status is None or component_status == 0:
        return None
    if component_status == SIGPIPE_STATUS:
        return None
    return masked_note(component_status)


def masked_note(component_status: int) -> str:
    """The one loud line. Names the status, and names its own reach.

    `app/tool_quality.py` matches `MASKED_NOTE_PREFIX`, so this text and the
    counter are one definition; the em-dash clause stating the blind spot is part
    of the contract, not decoration — a reader who sees only this line must be
    able to tell that a clean reading is not the same as a covered command.
    """
    return (f"{MASKED_NOTE_PREFIX} the FIRST component of the last pipeline "
            f"exited {component_status} while the command reported exit code 0 — "
            f"bash reports only the LAST component's status, so this failure is "
            f"otherwise invisible. Reach: the first component of the LAST "
            f"pipeline only. A mid-stage failure, an earlier pipeline in the same "
            f"command, and a producer killed by SIGPIPE (141) are NOT detected.]")


def is_masked_result(text: str) -> bool:
    """True when a persisted Bash result carries the note.

    A substring test rather than a parse of the status: the sweep reads text
    another process wrote, and the only fact it needs is that the tool fired.
    """
    return MASKED_NOTE_PREFIX in text
