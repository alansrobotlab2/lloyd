#!/usr/bin/env python3
"""Date the daily-note alarm blocks that a live re-read refutes (#2221, #3277).

Why a script and not the guardian
--------------------------------
The go-forward half of #2221 is in `agent-services/guardian/guardian.py`: the two families
that actually fire now write a coalesced section and seal it with a `cleared:` line when the
condition stops. That half cannot reach the blocks already in the vault, because
`Notifier.resolve` scans `DAILY_SCAN_DAYS = 3` notes (`notify.py:78`) — an alarm older than
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

Two witnesses, and why the second one exists
--------------------------------------------
A refutation needs a tool that can answer the question the block asserted. #2221's is
`supervisorctl status`, and it is structurally blind to every other family: `STATE_LINE`
looks for a `name: STATE` line, and the blocks raised by `tmpwatch` contain no supervisor
state string at all, so `refutations()` returns nothing for them whatever the supervisor
says — measured 2026-10-24, the sweep's own `count_refutations` returned 0 for
`memory/2026-10-06.md`, `2026-10-15.md` and `2026-10-16.md` and 0 corpus-wide while seven
unsealed `/tmp is running out of inodes` blocks sat in those three files. They are also the
shape `Guardian.reconcile_stranded_alert` produces on purpose: it seals the incident and
returns `sealed-note-out-of-reach` (`guardian.py:2764`) when the note is outside
`notify.DAILY_SCAN_DAYS`, so a fresh seal re-accumulates a stranded block each time.

`tmpwatch` is the tool that family's figure belongs to, and it is re-readable:
`tmpwatch.measure` is the same `statvfs` probe that raised the alarm, and
`fraction < tmpwatch.CLEAR_FRACTION` is the same comparison `Tmpwatch.tick` applies when it
resolves an incident. So #3277 adds a second refuter with that witness, bound to
`tmpwatch.ALERT_TITLE` alone. It dates from the reading and never from the figure the body
quotes — the bodies say 81% and 82% while the live reading on 2026-10-25 was 0.2769 against
a 0.70 clear line — and it refuses when the reading is at or over the line or cannot be had,
leaving the bytes alone. Like `read_running`'s `runner=`, the probe is injected
(`measurer=`, and `fraction_reader=` on `main`) because the gate runs where `/tmp` cannot be
steered; the default is the live probe, so a hand-run has no seam to get wrong.

What it deliberately does not do
--------------------------------
It never edits a block that already carries `cleared:` or `dated:`, so a second run writes
nothing. It never touches a block whose subject no tool can answer for: a `Rolled back …`
notice and a self-test notice have no live witness at all, and a disk figure with no probe
behind it is not refuted by a probe measuring a different filesystem — each refuter is bound
to the family that owns its reading, and a block matching neither comes back byte-identical.
And it does not delete or reword the body: a note that recorded a real outage keeps the
record; what stops is the standing instruction below it.
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
import tmpwatch as tw  # noqa: E402  (ALERT_TITLE / CLEAR_FRACTION / measure — the second witness)
from app.data_root import vault_root

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

#: The second refuter's stamp (#3277), and the four names are what make it checkable rather
#: than reassuring: the tool (`tmpwatch.measure` over `statvfs`), the value it returned, the
#: threshold that value was compared against, and the day the read happened. A reader with
#: those can re-run the probe and disagree; a reader with "looks fine now" cannot. The
#: threshold is the producer's own `CLEAR_FRACTION`, named as such, because the honest claim
#: is "below the line that closes this family's incident", not "low".
#:
#: The same two properties `REFUTED_LINE` is held to, for the same two reasons: apart from
#: its own `dated:` prefix it carries neither marker word (#1988 — `grep -c 'cleared:'` is
#: the instrument over these notes, so a stamp that used the word in prose would be counted
#: as a recovery nobody watched), and `dated:` appears exactly once, so one pass is not read
#: as two. "at or over" is not a hedge: the comparison is exclusive, so a wrong stamp implies
#: a reading at or above the threshold, which is the check the closing sentence hands the
#: reader.
TMP_REFUTED_LINE = (
    "{prefix} {day}: `tmpwatch.measure` (stdlib `statvfs`) reads `{path}` at "
    "{fraction:.4f} of its budget ({fraction:.0%}), below `tmpwatch.CLEAR_FRACTION` = "
    "{threshold:.2f} — the same comparison `Tmpwatch.tick` applies when it resolves this "
    "family's incident — so the usage quoted above is not the usage now. The alarm above "
    "records what was true when it fired and is not a standing instruction; this block sits "
    "outside the `notify.DAILY_SCAN_DAYS = {scan}` daily notes `Notify.resolve` scans back, "
    "so the guardian's own retraction could not reach it and nobody watched the counter come "
    "down. That observed recovery is the one thing the guardian's retraction marker asserts, "
    "and a dating is the weaker claim that is true here. If this line is wrong the usage is "
    "at or over {threshold:.0%} as of now and the alert above is real — re-run `df -i "
    "{path}` before acting on it.")


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


def fraction_refutes(title: str, fraction: float | None) -> bool:
    """Whether a `/tmp` reading refutes this block — the criterion, without the stamp.

    Three ways this is False, and each answers a different way the sweep could tell a lie:

    * `fraction is None` — no witness, so nothing is known. None is what `main` passes when
      the probe refused; a `0.0` allowed through instead would read as "totally clean" and
      date every block in the corpus.
    * `tmpwatch.ALERT_TITLE not in title` — the reading is of `/tmp`, so it refutes only the
      family that measured `/tmp`. This binding is what keeps a second refuter from becoming
      a general "some number is low today" pass over every figure in the vault.
    * `fraction >= tw.CLEAR_FRACTION` — the comparison is the exclusive one `Tmpwatch.tick`
      applies when it resolves an incident, so a reading of exactly the threshold refutes
      nothing. This is the branch that stops the sweep dating a block off the 81% its own
      body quotes: the refuter never reads the body for its number.
    """
    return (fraction is not None and tw.ALERT_TITLE in title
            and fraction < tw.CLEAR_FRACTION)


def fraction_stamp(title: str, day: str, fraction: float | None) -> str | None:
    """The stamp a refuted `/tmp` block earns, or None when the criterion does not hold.

    The reason the CLI reports is built from the same four numbers the line carries, so the
    printed pass and the written note cannot disagree about what was measured.
    """
    if not fraction_refutes(title, fraction):
        return None
    return TMP_REFUTED_LINE.format(prefix=notify.DAILY_DATED_PREFIX, day=day, path=tw.PATH,
                                   fraction=fraction, threshold=tw.CLEAR_FRACTION,
                                   scan=notify.DAILY_SCAN_DAYS)


def block_is_refuted(title: str, body: str, *, running: set[str],
                     fraction: float | None) -> bool:
    """Whether either witness refutes this block. The one criterion the count and the write
    both consult: `count_refutations` is the CLI's "how many blocks does this pass change",
    and two definitions of "refuted" is how a report starts counting blocks that the writer
    then leaves alone — or the other way round, which is worse.
    """
    if already_sealed(body):
        return False
    return bool(refutations(body, running)) or fraction_refutes(title, fraction)


def count_refutations(text: str, running: set[str], fraction: float | None = None) -> int:
    """How many unsealed guardian blocks in `text` a live read refutes.

    The sweep's own metric, exposed so the CLI's "0 hits" and the test's "0 hits" are one
    measurement and not two definitions that can drift apart: it asks the same question
    `date_note` asks, over the same body slices. `fraction` defaults to None — a caller that
    has not read `/tmp` is not a caller that read it as clean, so the second refuter
    contributes nothing until a reading is handed to it.
    """
    return sum(1 for title, start, end in _guardian_sections(text)
               if block_is_refuted(title, text[start:end], running=running,
                                   fraction=fraction))


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


def date_note(text: str, running: set[str], day: str,
              fraction: float | None = None) -> "tuple[str, int, list[str]]":
    """Append a `dated:` line to every refuted, unsealed guardian block: (text, n, notes).

    Appended at the END of the block, after the body, because the standing instruction these
    blocks finish with — "this needs a human" — is the last thing a reader sees before
    deciding to act, and a correction placed above it reads as part of the alarm. Appended
    rather than sealed in place because a block written before #2221 carries no open marker
    to replace, which is the same reason `resolve` cannot reach it.

    A block earns one stamp per witness that refutes it, which today means at most two, and
    only when the block asserts both a supervisor state and a `/tmp` figure — no family
    raises both. The count `n` is BLOCKS, not stamps: a block naming two programs already
    got two lines out of one pass before #3277, and the CLI's total is the number of notes
    a reader has to re-read, not the number of lines they land on.
    """
    inserts: list[tuple[int, str]] = []
    notes: list[str] = []
    for title, start, end in _guardian_sections(text):
        body = text[start:end]
        if not block_is_refuted(title, body, running=running, fraction=fraction):
            continue
        stamps: list[str] = []
        reasons: list[str] = []
        for program, state in refutations(body, running):
            stamps.append(REFUTED_LINE.format(prefix=notify.DAILY_DATED_PREFIX, day=day,
                                              program=program, state=state))
            reasons.append(f"{program}={state}")
        tmp_stamp = fraction_stamp(title, day, fraction)
        if tmp_stamp:
            stamps.append(tmp_stamp)
            reasons.append(f"measured fraction {fraction:.4f} < {tw.CLEAR_FRACTION:.2f}")
        inserts.append((end, separator(body) + "\n\n".join(stamps) + "\n"))
        notes.append(f"{title}: " + "; ".join(reasons))
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


def read_fraction(path: str = tw.PATH, *, measurer=tw.measure) -> float:
    """Ask tmpwatch's own probe what `/tmp` is at now. The second witness's `read_running`.

    `measurer` is `tmpwatch.measure`, so the reading this refuter acts on is the reading the
    alarm was raised from and the one `Tmpwatch.tick` resolves against — a second statvfs
    inlined here would be a second definition of "full" that could drift from the producer's
    own. It is a parameter for the reason `runner=` is one, and the same lesson attached:
    the gate runs the suite where no `/tmp` can be steered to a chosen percentage, so a node
    that shelled out could assert the at-or-above and failed-read refusals only on some
    machines. That is #2221's two clause-5 nodes coming back red at `cb235020` again, in the
    node written to prevent it.

    Two refusals, both a RuntimeError for `main` to turn into an exit code rather than a
    traceback. An exception out of the measurer (no `/tmp` at all, a statvfs the sandbox
    refused) is a read that did not happen; so is a value that cannot be a usage fraction.
    The second matters more here than it does for the supervisor table: `measure` returning
    a NaN or a 0.0 because the device was absent is, under `< CLEAR_FRACTION`, an instruction
    to date every tmpwatch block in the vault. A number out of `[0.0, 1.0]` therefore stops
    the pass instead of answering it.
    """
    try:
        fraction = float(measurer(path).fraction)
    except Exception as exc:                        # noqa: BLE001 — see read_running
        raise RuntimeError(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc
    if not 0.0 <= fraction <= 1.0:
        raise RuntimeError(f"{path} gave fraction {fraction!r}, which is not a usage "
                           f"fraction; nothing can be dated from it")
    return fraction


def main(argv: list[str] | None = None, *, running_reader=read_running,
         fraction_reader=read_fraction) -> int:
    """The CLI. Both witnesses are read here and nowhere else, and 2 means a pass was partial.

    `running_reader` and `fraction_reader` exist so a test can put numbers under this
    function without steering the box — the `runner=` reason one level up, applied to the
    boundary where the counts a human acts on are printed. The defaults are the live reads,
    so a hand-run reaches `supervisorctl` and `statvfs` with no seam to get wrong, and a
    refused read still stops the thing loudly: exit 2 with one line on stderr.

    A failed read exits 2 even when the other witness did write, and the reason is the
    report's own ambiguity. Seven stranded blocks and zero blocks both print "blocks: 0";
    the exit code is the only thing that separates "I looked and found nothing" from "I could
    not look". Whether bytes moved is in the printed count, which is why 2 does not also have
    to mean "nothing was written".
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notes", default=str(vault_root() / "memory"),
                    help="directory of dated daily notes")
    ap.add_argument("--conf",
                    default=str(REPO / "agent-services/supervisor/supervisord.conf"))
    ap.add_argument("--apply", action="store_true", help="write the dated: lines")
    ap.add_argument("--day", default=dt.date.today().isoformat())
    args = ap.parse_args(argv)

    try:
        running = running_reader(Path(args.conf))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    partial = False
    try:
        fraction: float | None = fraction_reader(tw.PATH)
    except Exception as exc:                        # noqa: BLE001 — a read, not a bug
        partial = True
        fraction = None
        print(f"{tw.ALERT_TITLE}: the `tmpwatch.measure` witness is unavailable, so no block "
              f"of that family can be dated this pass: {exc}", file=sys.stderr)

    total = 0
    for path in sorted(Path(args.notes).glob("20*.md")):
        text = path.read_text(encoding="utf-8")
        if not count_refutations(text, running, fraction):
            continue
        new, n, notes = date_note(text, running, args.day, fraction)
        total += n
        for s in notes:
            print(f"{path.name}: {s}")
        if args.apply:
            path.write_text(new, encoding="utf-8")
    if fraction is None:
        witness = "`tmpwatch.measure` unread; no block of that family dateable"
    else:
        witness = (f"{tw.ALERT_TITLE} fraction {fraction:.4f}"
                   f"{'<' if fraction < tw.CLEAR_FRACTION else '>='}{tw.CLEAR_FRACTION:.2f}")
    print(f"{'dated' if args.apply else 'refutable'} blocks: {total} "
          f"(RUNNING: {', '.join(sorted(running))}; {witness})")
    return 2 if partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
