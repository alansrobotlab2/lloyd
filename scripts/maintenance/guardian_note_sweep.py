#!/usr/bin/env python3
"""Date the daily-note alarm blocks that a live re-read refutes (#2221, #3277, #3502).

Why a script and not the guardian
--------------------------------
The go-forward half of #2221 is in `agent-services/guardian/guardian.py`: the two families
that actually fire now write a coalesced section and seal it with a `cleared:` line when the
condition stops. That half cannot reach the blocks already in the vault, because
`Notifier.resolve` scans `DAILY_SCAN_DAYS = 3` notes (`notify.py:103`) — an alarm older than
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

Three witnesses, and why the third one exists
---------------------------------------------
A refutation needs a tool that can answer the question the block asserted. #2221's is
`supervisorctl status`, and it is structurally blind to every other family: `STATE_LINE`
looks for a `name: STATE` line, and the blocks raised by `tmpwatch` contain no supervisor
state string at all, so `refutations()` returns nothing for them whatever the supervisor
says — measured 2026-10-24, the sweep's own `count_refutations` returned 0 for
`memory/2026-10-06.md`, `2026-10-15.md` and `2026-10-16.md` and 0 corpus-wide while seven
unsealed `/tmp is running out of inodes` blocks sat in those three files. They are also the
shape `Guardian.reconcile_stranded_alert` produces on purpose: it seals the incident and
returns `sealed-note-out-of-reach` (`guardian.py:2781`) when the note is outside
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

The third one is `guardian.tripwire_is_clear`, and it exists because the two tripwire blocks
have no owner at all once another alert takes `ALERT.md`'s single slot. `Guardian` answers
`Vault mass deletion — sync stopped` and `Lloyd data root damaged — promotions halted` with
the watch's own pair of reads — marker absent, and the tree evaluating clean against that
watch's baseline — and `_retract_tripwire_alert`'s docstring says of a displaced title that
"this process has no index to the stale section, which is the sweep's subject, not this
one's" (`guardian.py:2721-2722`). `memory/2026-10-20.md:17` is that case: the data-root halt fired
on 2026-10-20, a human cleared the marker, the title was displaced, and the block stayed a
standing instruction to restore a tree that measured whole — closed only by a hand-written
correction on 2026-10-26. Neither earlier witness can see it: `refutations()` needs a
`name: STATE` line, and `fraction_refutes` is bound to `/tmp`. So #3502 adds a third refuter
that asks **the guardian's own function**, lifted out of `Guardian` for exactly that reason —
`vaultwatch.sync_gate` and `datawatch.snapshot_gate` are NOT the same question, because both
answer True when the watch holds no baseline. The reading is one `vaultwatch.measure` per
root per pass (`tripwire_reader=`), because the data root is a 705,803-file walk, and the
stamp names the halt flag as a second measurement rather than assuming the halt was lifted:
the tree measuring whole is not the same fact as nobody holding a pause.

What it deliberately does not do
--------------------------------
It never edits a block that already carries `cleared:` or `dated:`, so a second run writes
nothing. It never touches a block whose subject no tool can answer for: a `Rolled back …`
notice and a self-test notice have no live witness at all, and a disk figure with no probe
behind it is not refuted by a probe measuring a different filesystem — each refuter is bound
to the family that owns its reading, and a block whose family no witness speaks for comes back
byte-identical. Four families are spoken for: a supervisor-state block by the live RUNNING
set, an inode block by `tmpwatch.measure`'s own fraction, a tripwire block by the
guardian's `tripwire_is_clear` over the watch that raised it, and a vault-snapshot block by
`vaultwatch.snapshot_age_state` — and each answers only the question its own reading asks,
so the third and fourth date a block and never clear one.
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
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "agent-services" / "guardian"))

import datawatch  # noqa: E402  (DataWatch — the data root's watch, the third witness)
import guardian as G  # noqa: E402  (tripwire_is_clear, VAULT/DATA_TRIP_ALERT_TITLE — the third witness)
import gstate  # noqa: E402  (AutomodState.halted — named as a measurement, never a precondition)
import notify  # noqa: E402  (DAILY_CLEARED_PREFIX / DAILY_DATED_PREFIX)
import policy  # noqa: E402  (VAULT_ROOT / DATA_ROOT / GUARDIAN_STATE / AUTOMOD_STATE)
import tmpwatch as tw  # noqa: E402  (ALERT_TITLE / CLEAR_FRACTION / measure — the second witness)
import vaultwatch  # noqa: E402  (VaultWatch / measure — the vault's watch, the third witness)
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
#: Where a section's OWN content stops: the first line of something that is not prose of that
#: block — a bare `---` thematic rule or an atx heading of any level.
#:
#: `sections()` ends a span at the NEXT `## ` heading, or at the end of the file for the last one,
#: so anything of this shape inside a span was swallowed by it. In the daily notes that tail is the
#: note's own `---` rule and the `### Session …` auto-captured digest that follows it: 37 of the 125
#: guardian spans of the 2026 notes have one, and 21 of those 37 are NOT the note's last section —
#: which is why this is a rule about the span and not an `is-last-section` test. Dating inside the
#: swallowed tail put a retraction below Alan's own digest, where the destructive instruction the
#: block retracts reads as unretracted to anyone reading the block (#3993).
SPAN_TAIL = re.compile(r"(?m)^(?:---[ \t]*$|#{1,6}[ \t])")
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


# ── the third witness: the guardian's own tripwire clear condition (#3502) ─────────

#: The two titles whose clear condition is a watch's pair of reads, mapped to the family
#: of that watch. Bound by title for the same reason `tmpwatch.ALERT_TITLE` is: a block
#: quoting a file count that this watch does not own must stay byte-identical.
TRIP_ALERT_TITLES: dict[str, str] = {
    G.VAULT_TRIP_ALERT_TITLE: "vault",
    G.DATA_TRIP_ALERT_TITLE: "data",
}

TRIP_REFUTED_LINE = (
    "{prefix} {day}: `guardian.tripwire_is_clear` reads `{root}` at {total} files against "
    "that watch's own baseline with its {family} tripwire marker absent — the same pair of "
    "reads `Guardian._retract_tripwire_alert` seals a live incident with — so the "
    "restore-and-swap instructions above are stale. `gstate.AutomodState` reports the "
    "promotion halt `{halt_path}` {halt_state} at the same read. This says the tree "
    "measures whole; it does not assert that the worker pause or the promotion halt was "
    "lifted, which is still whoever holds it. The alarm above records what was true when it "
    "fired and is not a standing instruction; this block sits outside the "
    "`notify.DAILY_SCAN_DAYS = {scan}` daily notes `Notifier.resolve` scans back, and once "
    "another alert held ALERT.md's single slot the guardian had no index to this section at "
    "all. If this line is wrong then the root is damaged again or the marker is set as of "
    "now: re-run `python3 agent-services/guardian/{watch_script}.py status` before acting on "
    "the text above.")


class TripwireReading(NamedTuple):
    """One tripwire family as the guardian's own condition read it, once per pass."""
    family: str          # "vault" | "data", the watch that raised the alert
    root: str
    total: "int | None"  # files the single walk returned; None means it returned nothing
    clear: bool          # `guardian.tripwire_is_clear`'s answer, unmodified
    why: str             # that function's own sentence, for the report and the refusals
    halt_path: str       # the flag file the halt state was read from
    halted: bool         # `AutomodState.is_halted()` at the same instant


def promotion_halt(*, gdir: Path | None = None) -> "tuple[str, bool]":
    """`(path, set?)` of the promotion halt flag, read the guardian reads it.

    Read as a MEASUREMENT for the stamp, never as a precondition of it: a tree that
    measures whole is not a lifted halt, and the guardian that halts promotions is not the
    one that clears them (`gstate.set_halted` has no matching timeout). The path is returned
    with the answer because `AutomodState.is_halted()` is `self.halted.exists()`, which
    answers False for a flag it cannot stat — a reader who wants to know whether "not set"
    was a clean read or an unreadable one needs to know WHICH file was read, which is why the
    stamp names it.
    """
    state = gstate.AutomodState(Path(gdir or policy.AUTOMOD_STATE))
    return str(state.halted), state.is_halted()


def tripwire_watches(*, gdir: Path | None = None) -> dict[str, object]:
    """The two watches the guardian runs, over the roots it watches.

    Constructing one only loads its persisted baseline (`VaultWatch._load`), so this walks
    nothing; the walk is `read_tripwire`'s, and it happens once per root there.
    """
    state = Path(gdir or policy.GUARDIAN_STATE)
    return {"vault": vaultwatch.VaultWatch(str(policy.VAULT_ROOT), state),
            "data": datawatch.DataWatch(str(policy.DATA_ROOT), state)}


def read_tripwire(*, watches: dict[str, object] | None = None,
                  measure=vaultwatch.measure,
                  halt_reader=promotion_halt) -> dict[str, TripwireReading]:
    """Ask `guardian.tripwire_is_clear` once per tripwire family, on one walk per root.

    WHY THIS IS A SEAM INJECTED AT THE CLI LIKE `running_reader=` AND `fraction_reader=`.
    The default measures `policy.VAULT_ROOT` and `policy.DATA_ROOT`, and the data root is a
    705,803-file walk as of 2026-11-02, so a gate node that let it run would both spend
    minutes on it and judge a tree the node does not own. The default stays the live read so
    a hand-run has no seam to get wrong, which is the same trade the other two witnesses
    made.

    WHY THE SNAPSHOT IS PASSED IN RATHER THAN LEFT TO THE CONDITION. `tripwire_is_clear`
    measures the tree itself when given no `snap` — correct for the guardian's startup
    reconciliation, and a second full walk here. Passing the snapshot is what
    `_retract_tripwire_alert` already does (`guardian.py:2740`), so the criterion is the
    guardian's and the walk count is this pass's.

    `watches` is injectable because the honest test for the no-baseline refusal needs a
    watch whose history it controls; `measure` is injectable because the walk-once claim is
    only checkable by counting the calls.
    """
    halt_path, halted = halt_reader()
    out: dict[str, TripwireReading] = {}
    for family, watch in (watches if watches is not None else tripwire_watches()).items():
        snap = measure(watch.root, skip_top=watch.skip_top)
        clear, why = G.tripwire_is_clear(watch, snap)
        out[family] = TripwireReading(
            family=family, root=str(watch.root),
            total=(None if snap is None else snap.total),
            clear=bool(clear), why=str(why),
            halt_path=halt_path, halted=bool(halted))
    return out


def tripwire_reading_for(readings: "dict[str, TripwireReading] | None",
                         title: str) -> "TripwireReading | None":
    """This block's tripwire family, or None when the title names neither.

    A family with no reading — the pass could not build that watch, or was injected a dict
    holding only the other — is a missing comparison, so it comes back None and the block is
    left alone exactly as a `clear=False` one is.
    """
    if not readings:
        return None
    for alert, family in TRIP_ALERT_TITLES.items():
        if alert in title:
            return readings.get(family)
    return None


def tripwire_refutes(title: str, readings: "dict[str, TripwireReading] | None") -> bool:
    """True only for a tripwire block the guardian's own condition answers clear."""
    reading = tripwire_reading_for(readings, title)
    return bool(reading and reading.clear)


# ── the fourth witness: the vault snapshot's own age reading (#3882) ─────────

#: The one title this witness speaks for, bound for the reason all three earlier
#: bindings exist: `snapshot_age_state` measures the vault's off-vault backup
#: repository, and a block whose subject is a file count, an inode figure or a
#: supervisor state is not refuted by a commit that happens to be recent.
SNAPSHOT_ALERT_TITLE = G.VAULT_SNAPSHOT_ALERT_TITLE

#: The reading's OWN sentence is quoted rather than re-rendered, and that is the
#: whole reason `SnapshotReading` carries `text`: `snapshot_age_state` already names
#: the repository, the commit and its age in the unit a person reads, and is the same
#: function `Guardian._vault_snapshot_stream` raises the alarm from, so a stamp that
#: built its own age string would be a second definition of an age that can drift
#: from the one the alarm and the guardian's own `cleared:` line print. It is checked
#: for the two marker words before it is written, because it is a rendered string
#: rather than this file's own prose and #1988's instrument is `grep -c 'cleared:'`
#: over these very notes.
SNAPSHOT_REFUTED_LINE = (
    "{prefix} {day}: `vaultwatch.snapshot_age_state` reads {quoted} with `fresh=True, "
    "owned=True` — the same reading `Guardian._vault_snapshot_stream` raises this alarm "
    "from, and the reading that leg now hands `Notifier.resolve` on a healthy pass — so "
    "the restore instructions above are stale. The alarm above records what was true "
    "when it fired and is not a standing instruction; this block sits outside the "
    "`notify.DAILY_SCAN_DAYS = {scan}` daily notes `Notifier.resolve` scans back, and "
    "until #3882 the leg that raised it had no now-healthy branch to seal it with at "
    "all, which is why a dating is the only retraction that can reach it here. That "
    "observed recovery is the one thing the guardian's retraction marker asserts, and "
    "the weaker claim is the true one. If this line is wrong the stream has stopped "
    "again as of now: re-run `python3 agent-services/guardian/vaultwatch.py status` "
    "before acting on the text above.")


class SnapshotReading(NamedTuple):
    """The vault's off-vault backup repository, as the alarm's own probe read it."""
    repo: str
    sha: "str | None"        # HEAD's commit, as `git log -1` answered it
    fresh: bool              # HEAD or the producer's beat inside the bound
    owned: bool              # the repo sits under `vaultwatch.snapshot_owner_dir()`
    max_age: float           # the bound those ages were weighed against, in seconds
    text: str                # the probe's own sentence, quoted into the stamp


def read_snapshot(*, age_reader=vaultwatch.snapshot_age_state) -> SnapshotReading:
    """Ask the probe that raises this alarm what the backup repository is at now.

    `age_reader` is a parameter for the reason `runner=`, `measurer=` and
    `tripwire_reader=` are, and the lesson is the same each time: the gate runs where a
    test cannot steer a real `git log` of the live off-vault repository to a chosen age,
    so a node that reached for it could observe the refusals only on some boxes. The
    default is the live probe, so a hand-run has no seam to get wrong.

    Two refusals, both a `RuntimeError` for `main` to turn into an exit code rather
    than a traceback. An exception out of the probe is a read that did not happen —
    `snapshot_age_state` shells out to `git`, and a sandbox that refuses a subprocess
    is the ordinary gate case. A response that is not the documented dict is the same
    thing with a friendlier face, and is refused rather than defaulted: a missing
    `fresh` key read as False would be correct by accident, and read as True would
    date every snapshot block in the vault off a value nobody measured.

    `fresh` and `owned` are carried separately and NOT folded, exactly as the producer
    folds them: an unowned repository's freshness is a fact about somebody's fixture,
    and it refutes nothing (#3149).
    """
    try:
        state = age_reader()
    except Exception as exc:                        # noqa: BLE001 — see read_running
        raise RuntimeError(f"cannot read the vault snapshot repository: "
                           f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(state, dict) or "fresh" not in state or "owned" not in state:
        raise RuntimeError(f"`vaultwatch.snapshot_age_state` gave {state!r}, which is not "
                           "the reading this witness needs; nothing can be dated from it")
    return SnapshotReading(repo=str(state.get("repo")), sha=state.get("sha"),
                           fresh=bool(state["fresh"]), owned=bool(state["owned"]),
                           max_age=float(state.get("max_age") or 0.0),
                           text=str(state.get("text") or ""))


def snapshot_refutes(title: str, reading: "SnapshotReading | None") -> bool:
    """Whether a snapshot reading refutes this block — the criterion, without the stamp.

    Three ways this is False, each answering a different way the sweep could lie:

    * `reading is None` — no witness, so nothing is known. `main` passes None when the
      probe refused; defaulting it to a fresh reading would date every snapshot block
      in the corpus from an absent probe.
    * `SNAPSHOT_ALERT_TITLE not in title` — the reading is of the vault's backup
      repository, so it refutes only the family that measured it. Without the binding
      this becomes a general "some repository is recent today" pass over the vault.
    * `not (reading.fresh and reading.owned)` — a stale HEAD is the ALARM, not its
      refutation, and an unowned repository is not the vault's copy whatever it holds.
      A block whose own alarm is still true must come back byte-identical.
    """
    return bool(reading and reading.fresh and reading.owned
                and SNAPSHOT_ALERT_TITLE in title)


def snapshot_stamp(reading: "SnapshotReading | None", day: str) -> "str | None":
    """The stamp a refuted snapshot block earns, or None when the criterion does not hold.

    The CLI reports the same reading this line quotes, so the printed pass and the
    written note cannot disagree about what was measured — the rule `fraction_stamp`
    follows. The quoted sentence is re-checked here rather than trusted: it arrives
    from a rendered string, and a probe whose wording changed must not be able to put
    the word `cleared:` onto a historical block and have #2221's own instrument count
    that as a retraction that never happened.
    """
    if not reading or not reading.text:
        return None
    quoted = reading.text.strip().rstrip(".")
    if (notify.DAILY_CLEARED_PREFIX in quoted or notify.DAILY_DATED_PREFIX in quoted
            or "`" in quoted):
        return None
    return SNAPSHOT_REFUTED_LINE.format(prefix=notify.DAILY_DATED_PREFIX, day=day,
                                        quoted=f"`{quoted}`",
                                        scan=notify.DAILY_SCAN_DAYS)


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


def trip_stamp(reading: TripwireReading, day: str) -> str:
    """The tripwire witness's stamp (#3502). `day` is the read date, the same `--day` value
    the other two witnesses stamp with — never a date lifted from the block, and never the
    date the alarm fired.

    What the line may claim is what the shared condition measured and nothing else: the root,
    the count that one walk returned, the absent marker, and the halt flag as a second
    reading. `guardian.tripwire_is_clear`'s own True sentence ends by refusing to assert that
    the pause or the halt was lifted, so neither does this.
    """
    return TRIP_REFUTED_LINE.format(
        prefix=notify.DAILY_DATED_PREFIX, day=day, family=reading.family,
        root=reading.root,
        total=("nothing" if reading.total is None else f"{reading.total:,}"),
        halt_path=reading.halt_path,
        halt_state=("SET" if reading.halted else "not set"),
        watch_script=f"{reading.family}watch",
        scan=notify.DAILY_SCAN_DAYS)


def block_is_refuted(title: str, body: str, *, running: set[str],
                     fraction: float | None = None,
                     tripwire: "dict[str, TripwireReading] | None" = None,
                     snapshot: "SnapshotReading | None" = None) -> bool:
    """Whether any witness refutes this block. The one criterion the count and the write
    both consult: `count_refutations` is the CLI's "how many blocks does this pass change",
    and two definitions of "refuted" is how a report starts counting blocks that the writer
    then leaves alone — or the other way round, which is worse.

    The four witnesses are disjoint by TITLE, which is what lets them OR together without
    one block collecting four stamps: the first answers only a `name: STATE` line, the
    second only `tmpwatch.ALERT_TITLE`, the third only the two titles in
    `TRIP_ALERT_TITLES`, the fourth only `SNAPSHOT_ALERT_TITLE`. Each of the optional
    readings defaults to None for the same reason — a pass that did not take a reading is
    not a pass that read it healthy.
    """
    if already_sealed(body):
        return False
    return (bool(refutations(body, running)) or fraction_refutes(title, fraction)
            or tripwire_refutes(title, tripwire) or snapshot_refutes(title, snapshot))


def count_refutations(text: str, running: set[str], fraction: float | None = None,
                      tripwire: "dict[str, TripwireReading] | None" = None,
                      snapshot: "SnapshotReading | None" = None) -> int:
    """How many unsealed guardian blocks in `text` a live read refutes.

    The sweep's own metric, exposed so the CLI's "0 hits" and the test's "0 hits" are one
    measurement and not two definitions that can drift apart: it asks the same question
    `date_note` asks, over the same body slices. `fraction` defaults to None — a caller that
    has not read `/tmp` is not a caller that read it as clean, so the second refuter
    contributes nothing until a reading is handed to it. Same for `tripwire` and `snapshot`.
    """
    return sum(1 for title, start, end in _guardian_sections(text)
               if block_is_refuted(title, text[start:end], running=running,
                                   fraction=fraction, tripwire=tripwire,
                                   snapshot=snapshot))


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


def stamp_position(text: str, start: int, end: int) -> tuple[int, str]:
    """(offset, padding) for the stamps of the block whose span is `text[start:end]`.

    `end` is the next `## ` heading, or the end of the file for the note's last section, so it is
    where the block's span stops — not where the block's writing stops. Splicing at `end` put the
    retraction below whatever the span had swallowed: in these notes that is the file's own `---`
    rule and the `### Session …` auto-captured digest after it, so on `memory/2026-11-04.md` the
    retraction of the block at :54-64 was written at :72, under Alan's digest at :68, and the
    destructive instruction it retracts still read as unretracted to anyone reading the block
    (#3993). The offset is therefore the end of the block's OWN content — the start of the blank run
    preceding the first thing in the span that is not prose — falling back to `end` when the span
    holds no such tail, which is the position the sweep has always used and the one that stays right
    for the 88 spans of 125 with no tail.

    The search starts after the block's own heading line, which is a `#`-heading like any other and
    must not end the block before it has begun. The offset stays inside `[start, end]`, which is what
    keeps a re-run a no-op: `already_sealed` and the cross-note backstop both read `text[start:end]`,
    so a stamp moved ABOVE the block's own heading or BELOW `end` would either date the block again
    every pass or be read as a foreign block's stamp and silently stop dating this note.

    `padding` is the second half of the answer: a `---` sitting directly under a text line is that
    line's setext heading, not a thematic rule, so where the swallowed rule had no blank line above
    it the stamp is followed by one — inserted, because reformatting the human's own lines is not
    this sweep's to do.
    """
    heading_end = text.find("\n", start, end)
    if heading_end < 0 or heading_end + 1 >= end:
        return end, ""
    tail = SPAN_TAIL.search(text, heading_end + 1, end)
    if tail is None:
        return end, ""
    cut = tail.start()
    # Back over the BLANK lines only — a blank line is a newline whose predecessor is also a
    # newline — so the block's own terminating newline stays with the block, and the stamp is
    # spliced after it with the note's own blank line as the separator below.
    while cut > heading_end + 1 and text[cut - 1] == "\n" and text[cut - 2] == "\n":
        cut -= 1
    return cut, "" if text[cut:tail.start()] else "\n"


def date_note(text: str, running: set[str], day: str, fraction: float | None = None,
              tripwire: "dict[str, TripwireReading] | None" = None,
              snapshot: "SnapshotReading | None" = None) -> "tuple[str, int, list[str]]":
    """Append a `dated:` line to every refuted, unsealed guardian block: (text, n, notes).

    Appended at the END of the block, after the body, because the standing instruction these
    blocks finish with — "this needs a human" — is the last thing a reader sees before
    deciding to act, and a correction placed above it reads as part of the alarm. Appended
    rather than sealed in place because a block written before #2221 carries no open marker
    to replace, which is the same reason `resolve` cannot reach it.

    A block earns one stamp per witness that refutes it. The four are bound to disjoint
    titles, so in practice a block earns one, and the count `n` is BLOCKS either way, not
    stamps: a block naming two programs already got two lines out of one pass before #3277,
    and the CLI's total is the number of notes a reader has to re-read, not the number of
    lines they land on.
    """
    inserts: list[tuple[int, str]] = []
    notes: list[str] = []
    for title, start, end in _guardian_sections(text):
        body = text[start:end]
        if not block_is_refuted(title, body, running=running, fraction=fraction,
                                tripwire=tripwire, snapshot=snapshot):
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
        reading = tripwire_reading_for(tripwire, title)
        if reading is not None and reading.clear:
            stamps.append(trip_stamp(reading, day))
            count = ("" if reading.total is None else f" ({reading.total:,} files)")
            reasons.append(f"{reading.family} tripwire clear over {reading.root}{count}")
        snap_stamp = snapshot_stamp(snapshot, day) if snapshot_refutes(title, snapshot) else None
        if snap_stamp:
            stamps.append(snap_stamp)
            reasons.append(f"snapshot repo {snapshot.repo} fresh at commit "
                           f"{(snapshot.sha or 'unknown')[:8]}")
        # The offset is the end of the block's CONTENT, not of its span: `end` is the next
        # `## ` heading, or the end of the file for the note's last section, and dating at
        # that put a retraction below the human's own `---` rule and session digest (#3993).
        at, pad = stamp_position(text, start, end)
        # `separator` reads the END OF THE BLOCK, not of the span: the span can run on past the
        # block's last line through a swallowed rule and digest, and asking it how the SPAN ends
        # would ask the wrong question about how the gap above the stamp should look.
        inserts.append((at, separator(text[start:at]) + "\n\n".join(stamps) + "\n" + pad))
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
         fraction_reader=read_fraction,
         tripwire_reader=read_tripwire,
         snapshot_reader=read_snapshot) -> int:
    """The CLI. All four witnesses are read here and nowhere else, and 2 means a partial pass.

    `running_reader`, `fraction_reader`, `tripwire_reader` and `snapshot_reader` exist so a
    test can put numbers under this function without steering the box — the `runner=` reason
    one level up, applied to the boundary where the counts a human acts on are printed. The
    defaults are the live reads, so a hand-run reaches `supervisorctl`, `statvfs`, the two
    tripwire watches and `vaultwatch.snapshot_age_state` with no seam to get wrong, and a
    refused read still stops the thing loudly: exit 2 with one line on stderr.
    `tripwire_reader` is called exactly once per pass, before any file is opened, because its
    default walks `~/lloyd-data` — over 700,000 files — and a per-note call would multiply
    that walk by the size of the corpus; `snapshot_reader` is likewise once per pass, since
    its default shells out to `git log` of the backup repository.

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

    tripwire: dict[str, TripwireReading] = {}
    try:
        tripwire = tripwire_reader() or {}
    except Exception as exc:                        # noqa: BLE001 — a read, not a bug
        partial = True
        print("tripwire: the `guardian.tripwire_is_clear` witness is unavailable, so no "
              f"tripwire block can be dated this pass: {exc}", file=sys.stderr)
    for reading in sorted(tripwire.values(), key=lambda r: r.family):
        if not reading.clear:
            partial = True
            print(f"{reading.family} tripwire over {reading.root} is un-dateable this "
                  f"pass: {reading.why}", file=sys.stderr)

    # Read ONCE, before the walk, for the same reason `tripwire_reader` is read once: the
    # default shells out to `git log` of the vault's off-vault backup repository, and a
    # per-note call would pay a subprocess for every dated daily note in the corpus to
    # ask a question the corpus answers with one number.
    snapshot: "SnapshotReading | None" = None
    try:
        snapshot = snapshot_reader()
    except Exception as exc:                        # noqa: BLE001 — a read, not a bug
        partial = True
        print(f"{SNAPSHOT_ALERT_TITLE}: the `vaultwatch.snapshot_age_state` witness is "
              f"unavailable, so no snapshot block can be dated this pass: {exc}",
              file=sys.stderr)
    if snapshot is not None and not (snapshot.fresh and snapshot.owned):
        # A stale HEAD is the alarm this witness exists to refute, not its refutation, and
        # an unowned repository is not the vault's copy whatever its age — both leave the
        # block standing, and both are reported rather than passed over in silence, the way
        # an un-clear tripwire is.
        partial = True
        print(f"{SNAPSHOT_ALERT_TITLE} witness does NOT refute: {snapshot.text}",
              file=sys.stderr)

    total = 0
    for path in sorted(Path(args.notes).glob("20*.md")):
        text = path.read_text(encoding="utf-8")
        if not count_refutations(text, running, fraction, tripwire, snapshot):
            continue
        new, n, notes = date_note(text, running, args.day, fraction, tripwire, snapshot)
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
    if not tripwire:
        witness += "; tripwire unread; no tripwire block dateable"
    else:
        for r in sorted(tripwire.values(), key=lambda x: x.family):
            witness += (
                f"; {r.family} tripwire {'clear' if r.clear else 'NOT clear'} over {r.root}"
                + ("" if r.total is None else f" ({r.total:,} files)")
                + ("" if r.clear else f" — {r.why}"))
    if snapshot is None:
        witness += "; snapshot unread; no snapshot block dateable"
    else:
        witness += (f"; snapshot {'fresh' if (snapshot.fresh and snapshot.owned) else 'NOT fresh'}"
                    + ("" if snapshot.sha else " (no HEAD)")
                    + ("" if snapshot.sha is None else f" at {snapshot.sha[:8]}")
                    + ("" if snapshot.owned else f" — repo {snapshot.repo} is NOT the vault's"))
    print(f"{'dated' if args.apply else 'refutable'} blocks: {total} "
          f"(RUNNING: {', '.join(sorted(running))}; {witness})")
    return 2 if partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
