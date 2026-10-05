#!/usr/bin/env python3
"""Retract or date the daily-note alarm blocks that a live supervisor read refutes (#2221).

Why a script and not the guardian
--------------------------------
The go-forward half of #2221 is in `agent-services/guardian/guardian.py`: the two families
that actually fire now write a coalesced section and seal it with a `cleared:` line when the
condition stops. That half cannot reach the blocks already in the vault, because
`Notifier.resolve` scans `DAILY_SCAN_DAYS = 3` notes (`notify.py:70`) — an alarm older than
three days is reported, never rewritten. So the uncleared blocks that assert a non-RUNNING
supervisor state for a program `supervisorctl status` reports RUNNING today — measured
2026-10-05: 15 of them, `memory/2026-09-06.md` through `memory/2026-09-29.md`, all naming
`lloyd-mc:lloyd-backend` — are reachable only by a pass over the notes themselves.

Why `dated:` and not `cleared:`
-------------------------------
`cleared:` is a claim about an observation: the guardian watched the condition stop. For a
block written on 2026-09-06 nobody watched anything recover, and writing `cleared:` would
manufacture that observation — the exact class of false all-clear this item is about. The
honest sentence is the weaker one: a later read refutes the state the body quotes, and the
retraction is dated to that read. `notify.DAILY_DATED_PREFIX` is the single definition of
the marker, so a reader and any later code see one word for each of the two facts.

What it deliberately does not do
--------------------------------
It never edits a block that already carries `cleared:` or `dated:`, so a second run writes
nothing. It never touches a block whose subject the supervisor cannot answer for (a
`Rolled back …` notice, a self-test notice, tmpwatch's disk figures): the refutation is
specific to a state string that supervisord owns and now contradicts. And it does not delete
or reword the body — a note that recorded a real outage keeps the record; what stops is the
standing instruction below it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "agent-services" / "guardian"))

import notify  # noqa: E402  (DAILY_CLEARED_PREFIX / DAILY_DATED_PREFIX)

#: The supervisor states that read as "this program is not up". `RUNNING` is absent by
#: construction; `STOPPED` is present because a body quoting it is asserting the program is
#: down, which is what a later `RUNNING` contradicts.
STATES = ("FATAL", "STOPPED", "EXITED", "BACKOFF", "STARTING")

#: One body line asserting a state about one program. The non-greedy `\\S+?` plus the
#: required `\\s+` is what makes the match correct rather than tidy: watched names carry
#: colons themselves (`lloyd-mc:lloyd-backend`), so splitting on `:` yields `lloyd-mc`.
STATE_LINE = re.compile(r"(?m)^(\S+?):\s+(%s)\b" % "|".join(STATES))
HEADING = re.compile(r"(?m)^## (.*)$")
GUARDIAN_TITLE = "Self-mod guardian:"
# Two properties this line has to keep, and both are measured by the corpus it fixes rather
# than by taste. It names the DATE and the read, because a refutation is a session and not a
# second. And apart from its own `dated:` prefix it contains NEITHER marker word: #2221's
# instrument is `grep -c 'cleared:'` over the dated notes, so a stamp that used that word in
# prose would be counted as a retraction that never happened — an instrument reading its own
# description, which is #1988's defect wearing a new hat. `dated:` appears once for the same
# reason: a second mention would make one pass look like two.
REFUTED_LINE = ("{prefix} {day}: `supervisorctl status` reports `{program}` RUNNING, so the "
                "`{state}` state quoted above is not the state now. The alarm above records "
                "what was true when it fired and is not a standing instruction; this block "
                "predates the retraction #2221 gives its alert family, so nobody watched the "
                "condition go away, and that observed recovery is the one thing the "
                "guardian's own retractions marker asserts. A dating is the weaker claim "
                "that is true here. If this line is wrong the program is down again as of "
                "now and the alert above is real — re-run `supervisorctl -c "
                "~/lloyd/agent-services/supervisor/supervisord.conf status` before acting on "
                "it.")


def running_programs(status_stdout: str) -> set[str]:
    """The programs `supervisorctl status` reports RUNNING."""
    out: set[str] = set()
    for line in status_stdout.splitlines():
        cols = line.split()
        if len(cols) >= 2 and cols[1] == "RUNNING":
            out.add(cols[0])
    return out


def refutations(body: str, running: set[str]) -> list[tuple[str, str]]:
    """The (program, state) pairs a body asserts that `running` contradicts.

    A block naming a program the supervisor does NOT report RUNNING is left alone, and that
    is the safety property: on a box where the service is genuinely down, this finds nothing
    to say. Matching is per line, so a body quoting two programs is refuted only for the one
    that came back.
    """
    return sorted({(name, state) for name, state in STATE_LINE.findall(body)
                   if name in running})


def already_sealed(body: str) -> bool:
    """Whether a retraction or a dating is already in the block.

    Substring rather than line-anchored on purpose: `resolve` writes the marker as the
    block's last line, and a dating is appended the same way, but a block whose BODY quotes
    the word `cleared:` in prose is not a block this sweep may rewrite — being conservative
    here costs one missed dating, and being loose costs an edit to a note that never asked
    for one.
    """
    return notify.DAILY_CLEARED_PREFIX in body or notify.DAILY_DATED_PREFIX in body


def sections(text: str):
    """Yield (title, heading_start, body_end) per `## ` section; end is exclusive.

    The heading start rather than the body start is returned because the rewrite needs to
    know where the previous section's bytes end, and one definition of the section boundary
    is what keeps a stamp from landing in the neighbouring block — the defect #1988 found in
    the sealing code, in the other direction.
    """
    found = list(HEADING.finditer(text))
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        yield m.group(1).strip(), m.start(), end


def count_refutations(text: str, running: set[str]) -> int:
    """How many unsealed guardian blocks in `text` a live read refutes.

    The sweep's own metric, exposed so the CLI's "0 hits" and the test's "0 hits" are one
    measurement and not two definitions that can drift apart: it asks the same two questions
    `date_note` asks, in the same order, over the same body slices.
    """
    return sum(1 for _title, start, end in _guardian_sections(text)
               if not already_sealed(text[start:end])
               and refutations(text[start:end], running))


def _guardian_sections(text: str):
    """(title, body_start, body_end) for each guardian section, heading excluded from body.

    Lives apart from the public helpers because `count_refutations` and `date_note` must
    agree on what "the body of one guardian block" is: if one included the heading line and
    the other did not, a state quoted in a title would be refuted by one and missed by the
    other, which is exactly the two-definitions defect the module docstring warns about.
    """
    for title, heading_start, end in sections(text):
        if GUARDIAN_TITLE not in title:
            continue
        yield title, heading_start + len("## " + title) + 1, end


def date_note(text: str, running: set[str], day: str) -> "tuple[str, int, list[str]]":
    """Append a `dated:` line to every refuted, unsealed guardian block: (text, n, notes).

    Appended at the END of the block, after the body, because the standing instruction these
    blocks finish with — "this needs a human" — is the last thing a reader sees before
    deciding to act, and a correction placed above it reads as part of the alarm. Appended
    rather than sealed in place because a block written before #2221 carries no open marker
    to replace, which is the same reason `resolve` cannot reach it.
    """
    inserts: list[tuple[int, str]] = []
    notes: list[str] = []
    for title, start, end in _guardian_sections(text):
        body = text[start:end]
        hits = [] if already_sealed(body) else refutations(body, running)
        if not hits:
            continue
        stamp = "\n\n".join(
            REFUTED_LINE.format(prefix=notify.DAILY_DATED_PREFIX, day=day, program=p, state=s)
            for p, s in hits)
        inserts.append((end, separator(body) + stamp + "\n"))
        notes.append(f"{title}: " + ", ".join(f"{p}={s}" for p, s in hits))
    if not inserts:
        return text, 0, []
    out = []
    cursor = 0
    for pos, addition in sorted(inserts):
        out.append(text[cursor:pos])
        out.append(addition)
        cursor = pos
    out.append(text[cursor:])
    return "".join(out), len(inserts), notes


def separator(body: str) -> str:
    """The whitespace to place before a stamp, given how the block currently ends.

    A block ending mid-line must not have its last word glued to the stamp, and one ending
    in blank lines must not gain a wider gap: the note's sections are separated by `## `, so
    a blank run the sweep invented is a formatting change it was never asked to make.
    """
    if body.endswith("\n\n"):
        return ""
    if body.endswith("\n"):
        return "\n"
    return "\n\n"


def read_running(conf: Path, *, runner=subprocess.run) -> set[str]:
    """Ask the live supervisor. Two failures are refused, and neither is a nonzero rc.

    `supervisorctl status` exits 3 when anything is not RUNNING, which is the ordinary
    answer on a box with a stopped unit; refusing on rc would blind the sweep exactly when
    the question matters. What is refused is an empty table, and a table in which nothing is
    RUNNING — with no RUNNING program there is nothing to refute a block with, and writing
    "no hits" from a failed read would report the whole corpus clean.

    `runner` is a parameter rather than a call on the module's own `subprocess` for one
    reason, and it is the reason #2221's two clause-5 nodes came back red at `cb235020`:
    the gate runs the suite where no supervisor is answering, so a node that shelled out
    could observe those two refusals only on a box that happens to have one — the same
    "green means nothing here" defect this file exists to fix, in the test. The properties
    are about parsing and refusal, which a stand-in answers identically on every box. This
    is not a seam for a test to assert an answer the box would not give: `main` calls it
    with the default runner, and a failed read there stops the apply loudly.
    """
    try:
        proc = runner(["supervisorctl", "-c", str(conf), "status"],
                      capture_output=True, text=True, timeout=30, check=False)
    except Exception as exc:                        # noqa: BLE001
        # `supervisorctl` not on PATH, the socket unit gone, the timeout tripped: all of
        # them arrive as an exception, and all of them are a read that did not happen. Left
        # uncaught the sweep dies with a traceback, which is honest but useless to a person
        # who asked "which blocks are refuted?"; converted to the same refusal the empty
        # table raises, `main` prints one line and exits 2 without writing anything.
        raise RuntimeError(f"cannot read {conf}: {type(exc).__name__}: {exc}") from exc
    if not (proc.stdout or "").strip():
        raise RuntimeError(f"supervisorctl gave no table (rc={proc.returncode}): "
                           f"{proc.stderr[:200]}")
    running = running_programs(proc.stdout)
    if not running:
        raise RuntimeError("supervisorctl reports nothing RUNNING; nothing can be dated")
    return running


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notes", default=str(Path.home() / "obsidian" / "memory"),
                    help="directory of dated daily notes")
    ap.add_argument("--conf",
                    default=str(REPO / "agent-services/supervisor/supervisord.conf"))
    ap.add_argument("--apply", action="store_true", help="write the dated: lines")
    ap.add_argument("--day", default=dt.date.today().isoformat())
    args = ap.parse_args(argv)

    try:
        running = read_running(Path(args.conf))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    total = 0
    for path in sorted(Path(args.notes).glob("20*.md")):
        text = path.read_text(encoding="utf-8")
        if not count_refutations(text, running):
            continue
        new, n, notes = date_note(text, running, args.day)
        total += n
        for s in notes:
            print(f"{path.name}: {s}")
        if args.apply:
            path.write_text(new, encoding="utf-8")
    print(f"{'dated' if args.apply else 'refutable'} blocks: {total} "
          f"(RUNNING: {', '.join(sorted(running))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
