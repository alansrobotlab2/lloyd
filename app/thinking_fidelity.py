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

What this module deliberately does NOT claim is the mechanism. The two
candidates in the item are engine-side state (MTP speculative decoding) and
decode-side prior. The evidence gathered in the item weighs against
cross-request bleed — the flagged traces are textually distinct from each other
and carry no content token from any other session — and for a decode-side prior
fired on a degenerate, taskless turn, reinforced by `preserve_thinking`
replaying a fabricated trace back as the model's own prior thought
(`app/harness/loop.py::_assistant_message_for_history`). Separating those two
is the engine-side A/B (`MTP_ENABLED=0`) and the operator's to run;
``scripts/thinking_replay_probe.py`` is the version that needs no engine
restart.

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

import json
import re
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
    """What one session file contributed to a `scan_store`."""
    path: Path
    blocks: int = 0
    flagged: int = 0
    exempt: int = 0
    unreadable: bool = False


@dataclass
class StoreScan:
    """The verdict `scan_store` reaches, with its own denominators."""
    root: Path
    files: int = 0
    unreadable: int = 0
    blocks: int = 0
    flagged: int = 0
    files_with_flags: int = 0
    blocks_exempt_meta: int = 0
    files_exempt_meta: int = 0
    flagged_files: list[str] = field(default_factory=list)

    def report(self) -> str:
        """The report, denominator first. See this module's docstring, bullet 2."""
        pct = (100.0 * self.flagged / self.blocks) if self.blocks else 0.0
        lines = [
            f"thinking-trace fidelity: flagged {self.flagged} of "
            f"{self.blocks} reasoning blocks scanned "
            f"({pct:.2f}% of blocks; {self.files} session files, "
            f"{self.files_with_flags} with a flagged block)",
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


def scan_file(path: Path) -> FileScan:
    """Scan one session file. A file that does not parse is reported, not skipped."""
    out = FileScan(path=path)
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError):
        out.unreadable = True
        return out
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return out
    return scan_messages(path, messages)


def scan_store(root: Path) -> StoreScan:
    """Scan every ``*.json`` session file under `root`.

    Read-only, and callers get it from `app.data_root.production_data_root()`
    rather than `app.paths.SESSIONS_DIR`: under the test suite `SESSIONS_DIR` is
    a scratch root with no sessions in it, which would make every live-store
    guard below report "0 of 0" and call it clean.
    """
    scan = StoreScan(root=root)
    for path in sorted(root.glob("*.json")):
        one = scan_file(path)
        scan.files += 1
        if one.unreadable:
            scan.unreadable += 1
            continue
        scan.blocks += one.blocks
        scan.flagged += one.flagged
        scan.blocks_exempt_meta += one.exempt
        if one.flagged:
            scan.files_with_flags += 1
            scan.flagged_files.append(path.name)
        if one.exempt:
            scan.files_exempt_meta += 1
    return scan
