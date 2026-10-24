#!/usr/bin/env python3
"""The falsifier leg the `Bash/logic` verdict never had: observe the Bash tool's
tool-side braced-reference substitution pass, through the tool itself (#3215).

    cd ~/lloyd && .venvs/lloyd/bin/python scripts/bash_tool_prepass_probe.py
    prepass=fired names=arr[0] rc=0

WHAT THIS EXISTS TO FIX
-----------------------
`knowledge/software/bash-tool-env-prepass.md` measures that the Bash tool resolves
every braced reference in the command string against **its own** environment before
any shell sees the string, so a braced subscript arrives empty while the unbraced
form reaches bash intact (`#2953`). The same note records the ledger consequence. As
re-counted 2026-10-24 over the 518 rows of
`~/lloyd-data/_pipeline/skills/reviews/verdicts.jsonl`, the `Bash/logic` key carries
29 rows, all 29 `rejected_unverifiable`, 27 of them opening `RE-CONFIRMED (step 0
SKIP)` — every one re-decided on an `evidence_cmd` that is an `ls -t` / `grep -m1`
pair over `candidate-bash-logic-*.md` plus a grep of one skill file. Neither leg
executes a braced reference, so each re-confirmation of a reason that includes "no
protocol failure exists" rests on evidence structurally incapable of holding a
counter-example, which is the class of defect this box files as a false green. This
script is the missing leg: one line on stdout, rc 0, that a
`scripts/skill_verdicts.py record --evidence-cmd` can store and re-execute.

It is an observation, not a guard: both `fired` and `absent` exit 0, because the
ledger refuses an rc-0 falsifier that declares nothing (#2052) and needs the row's
command to stay runnable whichever way the mechanism goes. Only a run that produced
**no reading at all** exits non-zero (see below), and it never prints `absent`.

WHY THROUGH `_bash` AND NEVER A BARE `bash -c`
----------------------------------------------
The suspect is the layer between the tool and the shell, not the shell. Spawning
`/bin/sh -c` by hand measures bash — which the 2026-10-14 note already blamed and
was retracted for, because `/bin/sh -> bash` on this box — so a bare `bash -c`
reproduction cannot fail for the reason being adjudicated and re-mints the same
blindness into a new row. Importing `agent_mcp.builtin_bash._bash` and handing it
the command string takes the measurement at the tool's own boundary: the bytes that
reach the spawned shell are the ones the pass left behind. Measured 2026-10-24, in
process, with no MCP host and no `bash -c`: the command below came back with the
emitter's line first and the braced expansion empty. That reading is also what
retired the note's claim that the pass "cannot be pinned by a unit test here …
`_bash()` never sees a braced reference to object to".

The emitter itself is NOT in Lloyd's Python — `grep -rn` for its two message texts
across `agent_mcp`, `app` and `workers` returns 0 on this tree (#3215's triage), and
the warning text arrives inside the *spawned command's* captured output. This probe
observes it; finding the owning layer is a different job.

THE READING, AND THE CONTROL INSIDE THE SAME COMMAND
----------------------------------------------------
One command, with two expansions of one array:

    arr=(prepasspresent); echo "prepassmarker=[ARR_BRACED] [ARR_UNBRACED]"

* **`fired`** — the emitter's own line names the reference (`Invalid environment
  variable name evaluates to an empty string: arr[0]`), or the marker came back
  empty with no line at all, which is the silent-consumption class the note
  documents for a name the tool's own environment happens to define. `names=`
  carries the names the pass printed, comma joined with its `", "` spacing collapsed
  so the token stays whitespace-free; `names=silent` is the signature-less case.
* **`absent`** — the marker holds what bash put there (`prepasspresent`) and nothing
  named the reference: the string reached bash intact, so no pass consumed it. This
  is the counter-example the stored verdict has never been able to print, and the
  reason the unbraced leg is not decoration — bash expands its own array whether or
  not the pass exists, so without a marker that differs between the two states an
  empty `[]` would be indistinguishable from an unset array.
* **no reading, exit 2, stderr only** — `_bash` returned a JSON error (bad cwd, no
  spawn, timeout), the shell was killed by a signal, or the positive control
  `[prepasspresent]` is missing so the command cannot be trusted. This state prints
  nothing on stdout precisely so a broken probe can never be filed as `absent`: an
  instrument whose only failure mode is "the mechanism is gone" is exactly how
  #3215's verdict stayed un-refutable through its 27 `RE-CONFIRMED` rows.

Nothing here spells the reference. It is assembled at runtime from `chr(36)`, and
this file carries no dollar-brace byte anywhere in its own source — a literal one
would be consumed by the pass being measured every time the file is authored through
the Bash tool, leaving a probe that asserts about a string that never reached disk.
`tests/test_bash_tool_prepass_probe.py` holds that as an assertion, not as advice.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Assembled at runtime, never spelled: see the last paragraph above.
BRACED_REF = chr(36) + "{arr[0]}"
UNBRACED_REF = "$arr"
ARR_NAME = "arr[0]"
SENTINEL = "prepasspresent"
MARKER = "prepassmarker"

# The emitter's two message classes: both are emitted before the command's first
# line of output, and each aggregates the names it consumed over the whole command
# string ("Invalid …" for names it cannot parse, "Referenced but unset …" for
# parseable names absent from its environment).
SIGNATURE_RE = re.compile(
    r"(?:Invalid environment variable name|Referenced but unset environment variable)"
    r" evaluates to an empty string:[ \t]*(?P<names>[^\n]+)")
MARKER_RE = re.compile(rf"{MARKER}=\[([^\]\n]*)\]")
# `_bash`'s own contract: raw mixed output, with `\n[exit code: N]` appended when
# the shell's rc is non-zero. Anchored at the end, so output that merely quotes the
# phrase cannot be read as an exit status.
EXIT_CODE_RE = re.compile(r"\[exit code: (-?\d+)\]\s*\Z")

NAMES_SILENT = "silent"
NAMES_NONE = "none"
NO_READING_EXIT = 2
LINE_RE = re.compile(r"^prepass=(?:fired|absent) names=\S+ rc=[0-9]+$")


@dataclass(frozen=True)
class Reading:
    """One of the states, reduced to what the ledger row will carry."""

    state: str  # "fired" | "absent" | "error"
    names: tuple[str, ...]
    rc: int | None
    detail: str = ""

    @property
    def names_token(self) -> str:
        """Whitespace-free, because the printed line is matched with `names=\\S+`."""
        return ",".join(self.names) if self.names else NAMES_NONE


def probe_command() -> str:
    """The one command, built so the two states cannot print the same bytes.

    `arr` is declared *inside* the command, so the unbraced leg proves bash ran the
    string it was handed and holds the array — which is what makes an empty braced
    marker an observation of the pass rather than of an unset variable. The double
    quotes are on purpose and protect nothing: the pass rewrites through quoting, a
    quoted heredoc delimiter and a backslash escape alike.
    """
    return f"arr=({SENTINEL}); echo \"{MARKER}=[{BRACED_REF}] [{UNBRACED_REF}]\""


def parse_names(text: str) -> tuple[str, ...]:
    """Every reference the pass named, in the order printed, de-duplicated."""
    names: list[str] = []
    for line in SIGNATURE_RE.finditer(text):
        for raw in line.group("names").split(","):
            name = raw.strip()
            if name and name not in names:
                names.append(name)
    return tuple(names)


def evaluate(bash_text: str | None) -> Reading:
    """Reduce what `_bash` returned to one reading. Never guesses at a state."""
    text = bash_text or ""

    payload = _json_error(text)
    if payload is not None:
        return Reading("error", (), None, f"_bash returned {payload!r}, so no shell ran")

    match = EXIT_CODE_RE.search(text)
    rc = int(match.group(1)) if match else 0
    if rc < 0:
        # Killed by a signal: `_bash` reports the negative sign, and a shell that
        # died never produced the expansion this probe reads.
        return Reading("error", (), rc, f"the shell was killed by signal {-rc}")

    marker = MARKER_RE.search(text)
    if marker is None:
        return Reading("error", (), rc, "the marker never printed, so the command did not run")
    if f"[{SENTINEL}]" not in text:
        return Reading("error", (), rc,
                       f"the positive control [{SENTINEL}] is absent, so the braced reading "
                       f"is unattributable to the substitution pass")

    names = parse_names(text)
    if ARR_NAME in names:
        return Reading("fired", names, rc)
    if marker.group(1).strip() != SENTINEL:
        # bash's own value did not reach the marker and nothing was named: the
        # reference was consumed without a signature, the silent class.
        return Reading("fired", (NAMES_SILENT,), rc)
    return Reading("absent", (NAMES_NONE,), rc)


def format_reading(reading: Reading) -> str:
    """The single stdout line. One format, shared by the probe and its tests."""
    return f"prepass={reading.state} names={reading.names_token} rc={reading.rc}"


def _json_error(text: str) -> Any:
    """The payload when `_bash` answered with one of its early-return JSON errors."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None
    return parsed.get("error") if isinstance(parsed, dict) and "error" in parsed else None


def main() -> int:
    # Imported here, not at module scope: the parse helpers above are the graded
    # half of this file and must stay importable in a checkout without `mcp`.
    from agent_mcp.builtin_bash import _bash

    reading = evaluate(asyncio.run(_bash({"command": probe_command()})))
    if reading.state == "error":
        # Nothing on stdout: an absent line is a failed probe, a printed line is a
        # reading, and the two must never be confusable.
        print(f"{Path(__file__).stem}: {reading.detail}", file=sys.stderr)
        return NO_READING_EXIT
    print(format_reading(reading))
    return 0


if __name__ == "__main__":
    sys.exit(main())
