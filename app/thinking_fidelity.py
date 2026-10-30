"""Fidelity of recorded reasoning: does this thinking trace describe this turn?

Backlog #1510. The primary emits reasoning blocks describing a request that
was never made. The canonical example is
``sessions/20260925_001902_iv029b.json``: user ``[0]`` asks what
``stream_chat`` does, ``[3]`` is a ``Read`` call, ``[5]`` is a 1,223-character
trace about *"the user asking me to reproduce my complete previous thinking
verbatim using the audit tool"*, and ``[6]`` answers the original question
correctly. No such request exists in that session, in any prior turn, or in the
injected context, and the answer looks fine from the response alone — the
defect is only visible in ``reasoning``.

Scope is a moving number and nothing here should read as current: the store is
live, and the sessions it counts include the sessions that run this check, so a
count written into prose is a snapshot of the second it was taken. ``scan_store``
measured one on 2026-09-25 — 23,325 reasoning blocks in 1,049 session files, 137
flagged in 76 files, 16 of those in 2 sessions whose own user turn names the
defect — and ``python -m scripts.thinking_fidelity_scan`` prints today's, with the
denominator beside the flagged count, which is the number to quote. What does not
move: every hit is ``model='primary'``, every hit sits on a ``role=\"thinking\"``
row, and onset is 2026-09-22 — a per-day scan shows zero on every earlier day the
store holds, and its gap from 2026-09-12 to 2026-09-22 is the data-root migration
of 2026-09-22 (``6426668b``), not an absence of sessions.

A session's stored form is not stable, and this module reads both of them.
``scripts/groundskeeper/retention-sweep.py::sweep_sessions`` gzips
``<id>.json`` to ``<id>.json.gz`` and unlinks the original on a policy that is
*gzip and never delete*, so an archive is the only copy a session will ever
have. Reading only ``*.json`` therefore made the nightly sweep a silent
denominator loss (#3590): the store's whole-store block count fell
280,864 → 260,014 across four runs of 2026-10-27…10-30 with nothing deleted,
and a day whose sessions had mostly been archived reported ``scanned: true,
flagged: 0`` — a clean-looking day that had not been measured. So
`session_paths` enumerates both forms, `scan_file` decodes an archive as
transparently as a plain file, and every `StoreScan` carries
`files_archived` naming how much of its own denominator came from one. That is
the read-side half of what `app/uptake.py` (#1848) already does for its own
transcripts; the retention sweep itself is deliberately untouched.

What this module deliberately does NOT claim is the mechanism. The two
candidates in the item are engine-side state (MTP speculative decoding) and
decode-side prior. The evidence gathered in the item weighs against
cross-request bleed — the flagged traces are textually distinct from each other
and carry no content token from any other session — and for a decode-side prior
fired on a turn that names no task, reinforced by `preserve_thinking`
replaying a fabricated trace back as the model's own prior thought
(`app/harness/loop.py::_assistant_message_for_history`). Separating those two
is the engine-side A/B (`MTP_ENABLED=0`) and the operator's to run;
``scripts/thinking_replay_probe.py`` is the version that needs no engine
restart. The turn it replays names no task, but it is not a system-block-only
request: it is the system prompt plus a one-word ``role="user"`` turn
(``FIXED_USER_TURN``), because vLLM answers a message list with no user turn
with ``400 "No user query found in messages"`` and never consults the model —
a probe shaped that way scores nothing and used to report that as a clean rate
(#2164).

Two decisions in here are load-bearing and are the reason this file exists
rather than a grep in a test:

  - **The marker set is a lexical signature, not a semantic judge.** It catches
    the shapes the traces actually use. It cannot catch a fabrication that
    invents some *other* conversation, and it fires on a session whose own
    subject is this defect (a distiller asked to triage #1510 reasons about
    "reproducing previous thinking" because that is what it is reading). The
    second failure is why ``names_the_defect`` exempts a whole session and
    ``StoreScan`` reports the exempt count on its own line rather than folding
    it into the flagged total or hiding it.
  - **Every report prints the denominator beside the flagged count.** A bare
    "0 flagged" is indistinguishable from "0 scanned", which is the failure
    mode this whole store has already produced twice (`app/uptake.py`).
"""

from __future__ import annotations

import gzip
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

#: The six lexical markers, transcribed from backlog #1510's acceptance clause.
#: Each is one shape the fabricated traces were observed to take; the set is
#: what was measured to separate the known corpus from the control corpus.
FABRICATION_MARKERS: tuple[str, ...] = (
    # "My previous turn had no substantive reasoning; I simply acknowledged…"
    r"previous thinking",
    # The CoT-extraction-refusal shape, with or without "complete"/"previous".
    r"reproduce\s+(?:my|your|the)\s+(?:complete\s+|previous\s+)?"
    r"(?:thinking|reasoning)",
    # "The user hasn't asked anything yet… There's no actual task."
    r"no actual\s+(?:task|question)",
    r"no substantive reasoning",
    # "…the last message is just a system reminder."
    r"just a system reminder",
    # The invented tool: no tool named `audit` exists in this harness. Bounded on
    # both sides on purpose — the bare substring also sits inside "audit tooling",
    # and a healthy trace about the audit tooling under `scripts/` would otherwise
    # be withheld from preserved thinking. Measured on the live store on
    # 2026-09-25 (23,666 blocks, `scripts/thinking_fidelity_scan.py`): the boundary
    # loses exactly one flagged block, in a session whose own user turn names the
    # defect and which is exempt anyway, so nothing non-exempt is lost.
    r"\baudit tool\b",
)

FABRICATION_RE = re.compile("|".join(f"(?:{m})" for m in FABRICATION_MARKERS),
                            re.IGNORECASE)


def matched_marker(text: str | None) -> str:
    """The marker text that fired, or ``""`` when none did.

    A caller that withholds or counts a block should say which marker fired:
    "audit tool" and "no actual task" are different shapes, and a log line that
    names only "flagged" cannot be checked against the marker list later. This is
    the single place the regex is run for a decision, and
    `flag_fabricated_reasoning` is its boolean face, so no caller can re-derive
    the verdict a second way.
    """
    if not text:
        return ""
    hit = FABRICATION_RE.search(text)
    return hit.group(0) if hit else ""


def flag_fabricated_reasoning(text: str | None) -> bool:
    """True when a recorded reasoning block carries the fabricated-trace signature.

    True means *"this trace is not trustworthy evidence for the turn that
    produced it"* — it is a lexical verdict, so a caller may drop the block from
    what the engine sees but must not delete the row that recorded it
    (`app/transcript_entries.py::build_thinking_entry` keeps writing it).
    """
    return bool(matched_marker(text))


def names_the_defect(text: str) -> bool:
    """True when a user turn itself talks about the defect.

    A session whose prompt quotes "reproduce my complete previous thinking" is
    almost always reading a trace about it — the distiller that found #1510, or
    this item's own triage turn. Every reasoning block in such a session is
    *about* the marker words by construction, so flagging them would be a false
    positive on the session, not on the model. Exempting it is a scope decision
    the check makes out loud: `StoreScan.blocks_exempt_meta` is reported beside
    the flagged count and never summed into it.
    """
    return bool(text) and bool(FABRICATION_RE.search(text))


# --------------------------------------------------------------- store scan --

#: The archived form of a transcript, and the two forms a store may hold one in.
#: `scripts/groundskeeper/retention-sweep.py::sweep_sessions` is what writes the
#: first. Spelled here rather than imported from `app.uptake.py`, which defines
#: the same literal for its own reads: the session store is `app/`'s layout, and
#: an absence check on this module (`git grep -n gz app/thinking_fidelity.py`)
#: has to resolve to the suffix it is asking about — a grep that only lands in
#: another module is how a reader concludes "no gzip handling here" and is right.
SESSION_ARCHIVE_SUFFIX = ".json.gz"
SESSION_SUFFIXES: tuple[str, ...] = (".json", SESSION_ARCHIVE_SUFFIX)


def session_id(name_or_path: str | Path) -> str:
    """The session a transcript file names, with either suffix.

    NOT `Path.stem`: for an archive that is ``<id>.json``, which is a different
    identity than the plain file's and makes `session_paths` treat one session
    as two.
    """
    name = (name_or_path.name if isinstance(name_or_path, Path)
            else str(name_or_path))
    for suffix in (SESSION_ARCHIVE_SUFFIX, ".json"):   # the longer one first
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def session_paths(root: Path) -> list[Path]:
    """Every session transcript under `root`, live or archived, one per session.

    The one walk, shared by `scan_store` and the per-day walk so neither can
    drift back to plain files alone. Three properties, each of which was a bug
    somewhere (#3590, and `app/uptake.py::_session_transcripts` for the same
    store):

      - both ``*.json`` and ``*.json.gz`` are enumerated, so archiving a session
        moves no denominator;
      - a session id appears EXACTLY ONCE, so no file is parsed twice. A store
        holding both ``X.json`` and ``X.json.gz`` is the state a sweep
        interrupted between its `gzip.open` and its `path.unlink` leaves behind;
        the LIVE file wins, being the current prose. Ids differing in more than
        the suffix stay separate files.
      - the result sorts by the SESSION ID and not the raw name, so the two forms
        of one session are adjacent and one day's ids stay contiguous — the
        date prefix leads in both forms, and that contiguity is the fact
        `scan_store_by_day` relies on to close a day when the day changes.
    """
    chosen: dict[str, Path] = {}
    for suffix in SESSION_SUFFIXES:                     # ".json" first, and it wins
        # `*.json` does not match a `.json.gz` name — fnmatch has no implicit
        # "anything" after a dot — so the two globs are disjoint and the
        # `setdefault` below only ever fires for the interrupted-sweep overlap.
        for path in root.glob(f"*{suffix}"):
            if path.is_file():
                chosen.setdefault(session_id(path.name), path)
    return sorted(chosen.values(), key=lambda p: session_id(p.name))


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def reasoning_blocks(messages: Iterable[dict]) -> Iterator[str]:
    """Every recorded reasoning string in one session's message list.

    Both carriers count: ``role="thinking"`` rows (the per-phase rows
    `app/transcript_entries.py::build_thinking_entry` writes) and assistant
    rows, which carry `reasoning` too. Reading only the assistant rows would see
    the phases the harness replays and miss the phases the router persisted, which
    is most of what a re-score has to work over.
    """
    for m in messages:
        if not isinstance(m, dict):
            continue
        r = m.get("reasoning")
        if isinstance(r, str) and r.strip():
            yield r


@dataclass
class FileScan:
    """What one session file contributed to a `scan_store`.

    `archived` records the form the file was READ in, not the shape of its bytes:
    a `.json.gz` that turns out to be undecodable is still an archived file, and
    a `files_archived` that quietly excluded the rotten archives would make the
    witness under-report exactly the files whose readability is in question.
    """
    path: Path
    blocks: int = 0
    flagged: int = 0
    exempt: int = 0
    unreadable: bool = False
    archived: bool = False


@dataclass
class StoreScan:
    """The verdict `scan_store` reaches, with its own denominators.

    `files_archived` is the denominator's witness (#3590): how much of `files`
    arrived through the gzip route. The retention sweep moves sessions between
    the two forms every night, and until this field existed a day could lose
    most of its denominator to that move and still print `scanned: true` with
    `flagged: 0`.
    """
    root: Path
    files: int = 0
    unreadable: int = 0
    blocks: int = 0
    flagged: int = 0
    files_with_flags: int = 0
    blocks_exempt_meta: int = 0
    files_exempt_meta: int = 0
    files_archived: int = 0
    flagged_files: list[str] = field(default_factory=list)

    def report(self) -> str:
        """The report, denominator first. See this module's docstring, bullet 2.

        The headline carries `files_archived` (#3590 clause 4) because the
        printed line and the JSONL row are read by different readers and have to
        be one measurement: a run whose prose names one file count while its row
        names another is two numbers, and the one in prose is the one a human
        quotes. No figure for the live store belongs in here — the store is
        growing and archiving while it is read, which is the whole reason the
        count is a field on a row rather than a sentence in a docstring.
        """
        pct = (100.0 * self.flagged / self.blocks) if self.blocks else 0.0
        lines = [
            f"thinking-trace fidelity: flagged {self.flagged} of "
            f"{self.blocks} reasoning blocks scanned "
            f"({pct:.2f}% of blocks; {self.files} session files, "
            f"{self.files_with_flags} with a flagged block, "
            f"{self.files_archived} archived)",
            f"  exempt as meta-session (the session's own user turn names the "
            f"defect, so its reasoning is about the marker words): "
            f"{self.blocks_exempt_meta} blocks in {self.files_exempt_meta} "
            f"file(s)",
        ]
        if self.unreadable:
            lines.append(f"  unreadable session files: {self.unreadable}")
        return "\n".join(lines)


def scan_messages(path: Path, messages: list) -> FileScan:
    """Score one session's already-parsed message list.

    The seam exists because the live store is being written while it is read: a
    caller that walks the store once and a check that opens each file again are
    two different stores, and on 2026-09-25 the difference between those two
    reads of `~/lloyd-data/sessions` was one block and three files — the session
    the checking round was writing at the time. `scan_file` is the ordinary
    route; this is the one that lets a comparison be exact about bytes already
    in hand.
    """
    out = FileScan(path=path)
    exempt = names_the_defect(" ".join(
        _text_of(m) for m in messages
        if isinstance(m, dict) and m.get("role") == "user"))
    for block in reasoning_blocks(messages):
        out.blocks += 1
        if flag_fabricated_reasoning(block):
            if exempt:
                out.exempt += 1
            else:
                out.flagged += 1
    return out


def load_session_doc(path: Path):
    """One parsed session document from a plain or gzipped transcript.

    Raises whatever the read fails with — the caller turns that into an
    `unreadable` `FileScan` — and is a function rather than inline because the
    exception set is the interesting part and it is not the plain file's set.
    `zlib.error` is the damaged-archive shape that is NOT an `OSError`: a valid
    10-byte gzip header over bytes that are not a deflate stream raises it out of
    the decompressor. `EOFError` is the truncated one. Both are named here for
    the reason `app/uptake.py::read_session` names them: a walk reads every
    transcript in one loop, and an escaping exception ends the walk, so one
    rotten member would cost the whole corpus — which is the opposite of what a
    denominator is for.
    """
    if path.name.endswith(SESSION_ARCHIVE_SUFFIX):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    return json.loads(path.read_text(errors="replace"))


def scan_file(path: Path) -> FileScan:
    """Scan one session file. A file that does not parse is reported, not skipped.

    Reads either stored form (#3590): the retention sweep gzips a session and
    unlinks the original, so a reader of plain files alone watches the
    denominator shrink with nothing deleted. The same document yields the same
    blocks, flags and exemptions whichever suffix it carries, and an archive that
    cannot be decoded lands in `unreadable` exactly as an unparseable plain file
    does — the gzip route inherits the never-dropped rule rather than getting a
    second one.
    """
    out = FileScan(path=path)
    archived = path.name.endswith(SESSION_ARCHIVE_SUFFIX)
    out.archived = archived
    try:
        data = load_session_doc(path)
    except (OSError, ValueError, EOFError, zlib.error):
        out.unreadable = True
        return out
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return out
    # `scan_messages` builds its own `FileScan`, so the form has to travel with the
    # result and not with `out`: setting `archived` on the discarded one is how
    # `files_archived` ends up blind to exactly the archives that DECODED — the
    # three rotten ones still reported, the readable one not, which is the witness
    # reporting worst on the files it was invented to count.
    scanned = scan_messages(path, messages)
    scanned.archived = archived
    return scanned


def accumulate_file(scan: StoreScan, one: FileScan) -> None:
    """Fold one file's contribution into a running `StoreScan`, in place.

    The one arithmetic, and the reason a store and a slice of it cannot disagree:
    every counter of every `StoreScan` this module produces moves through here,
    including the two a `--record --per-day` run builds from ONE walk
    (`scripts/thinking_fidelity_scan.scan_store_by_day`, #2819) — the whole store
    and each day's slice, fed by the same `FileScan` object. A tally re-implemented
    beside it can drift in any of the four counters, and the one it would drift in
    is `exempt`, which is the e206f304 meta-session rule the per-day series exists
    to measure.

    Takes the `FileScan` and not the path: a caller that has already paid for the
    read must not pay for it twice to change its mind about what to count.
    """
    scan.files += 1
    if one.archived:
        # BEFORE the unreadable return, on purpose: the witness is about which
        # files came off the gzip route, and a rotten archive is one of them. A
        # tally placed after it would report `files_archived: 0` on a store whose
        # archives all failed to decode — the exact run the witness exists to
        # make loud, and the same "unreadable is reported, never dropped" rule
        # the `unreadable` counter above is.
        scan.files_archived += 1
    if one.unreadable:
        scan.unreadable += 1
        return
    scan.blocks += one.blocks
    scan.flagged += one.flagged
    scan.blocks_exempt_meta += one.exempt
    if one.flagged:
        scan.files_with_flags += 1
        scan.flagged_files.append(one.path.name)
    if one.exempt:
        scan.files_exempt_meta += 1


def iter_file_scans(paths: Iterable[Path]) -> Iterator[FileScan]:
    """Read `paths` one at a time, yielding each file's `FileScan` in the order given.

    The read as a stream, so a caller can fold one pass into several tallies at
    once instead of walking the store once per tally (#2819: the day roll-up and
    the whole-store row used to be two walks of the same ~10,000 files, and every
    session file was JSON-parsed twice per nightly run). It is also the seam a
    counting wrapper goes on to prove a run reads N files and not 2N —
    `scan_files`, `scan_store` and the per-day walk all read through this name.

    Unreadable files are yielded as `unreadable` `FileScan`s, never dropped: what
    a missing file counts as is `accumulate_file`'s answer, not this generator's.
    """
    for path in paths:
        yield scan_file(path)


def scan_files(root: Path, paths: Iterable[Path]) -> StoreScan:
    """Scan exactly the session files `paths` names, attributing them to `root`.

    The slice seam. `scan_store` is this call over `session_paths` of a
    directory, and a caller that wants a *slice* of a store — one calendar day of
    it, which is what `scripts/thinking_fidelity_scan.py --per-day` groups by
    (#1770) — has to get its numbers through here or through `accumulate_file`,
    the same fold, rather than a tally of its own beside it.

    `root` is carried for the report, not for the walk: a day's scan says which
    store the day came from, and has no directory of its own.
    """
    scan = StoreScan(root=root)
    for one in iter_file_scans(paths):
        accumulate_file(scan, one)
    return scan


def scan_store(root: Path) -> StoreScan:
    """Scan every session file under `root`, live or archived.

    Read-only, and callers get it from `app.data_root.production_data_root()`
    rather than `app.paths.SESSIONS_DIR`: under the test suite `SESSIONS_DIR` is
    a scratch root with no sessions in it, which would make every live-store
    guard below report "0 of 0" and call it clean.

    Exactly `scan_files` over `session_paths`, so a slice taken through
    `scan_files` and a whole store taken through here are the same arithmetic,
    and so the walk every shape of run takes is `session_paths` and nothing else
    — the two globs this replaced (`*.json` here, `*.json` again in the per-day
    walk) could and did drift apart from each other and from the store (#3590).
    """
    return scan_files(root, session_paths(root))
