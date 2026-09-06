"""Background triage of the Lloyd backlog: re-check premises, retire the stale.

The backlog holds ~53 open items going back to February. Many were written
against a system that has since changed. This source works through them oldest
first and asks one question per item: **is this still true?**

It deliberately does **not** implement anything. Triage is read-only, and its
output is a verdict plus the evidence for it. Confirmed items accumulate as
*verified* work for the self-modification loop to pick up later, through the
normal gate. Splitting it this way is the whole point: the failure mode worth
avoiding is a confident, tested, gated change that solves a problem nobody has,
and that failure mode is only reachable if implementation can start from an
unverified premise.

Retiring an item is a success. For a backlog this age it is the *expected*
outcome, and a pipeline that only counts code as progress would quietly turn a
stale backlog into a pile of unnecessary changes.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.backlog-selfmod")

NAME = "backlog-selfmod"
DEFAULT_PRIORITY = 75
DEDUP_KEY = "backlog-selfmod:triage"

PROMPT = """\
You are triaging one item from Lloyd's own backlog. It was written {age} days \
ago, and the system has changed since. Your job is to find out whether it is \
**still true** — not to fix it.

<item id="{item_id}" status="{status}" priority="{priority}">
# {name}

{body}
</item>

Work in this order:

1. **State the premise.** In one sentence, what does this item assert is true \
about the system? If it asserts nothing checkable (it is an idea, a research \
prompt, or a wish), the verdict is `unverifiable`.
2. **Decide whether it is even about Lloyd's own code.** Hardware, robots, \
external services and vault content are `not_code`. Do not investigate further.
3. **Design a check.** A command, a file to read, a grep, a metric to query — \
something that would come out differently depending on whether the premise \
holds. Write it down before running it.
4. **Run the check.** Use Read, Grep, Glob and Bash. You are read-only: do not \
edit, write, or commit anything. Do not start a selfmod round.
5. **Reach a verdict** from the evidence:
   - `confirmed` — the premise still holds; the problem is real today
   - `already_done` — it was real, and something has since fixed it
   - `stale` — the premise no longer describes this system
   - `unverifiable` — the item states no claim that can be checked
   - `not_code` — not about Lloyd's own code

Rules that matter:

- **`stale` and `already_done` are good outcomes.** Most of a backlog this old \
should reach them. Do not strain to confirm an item so it looks productive.
- **Never guess.** If you could not run a conclusive check, say `unverifiable` \
and explain what you would need. A wrong `confirmed` sends the selfmod loop \
after a problem that does not exist.
- **Quote your evidence.** File paths with line numbers, command output, commit \
SHAs. A verdict without evidence is unusable, because the point of this pass is \
that a human can audit it later.
- If you find the premise confirmed, also state **how the fix would be \
verified** — the check you just ran should fail to reproduce afterwards.

Finish with exactly this block and nothing after it:

VERDICT: <one of confirmed|already_done|stale|unverifiable|not_code>
CHECK: <the command or method you ran, one line>
EVIDENCE: <2-4 sentences citing what you actually observed>
ACCEPTANCE: <if confirmed: what must become true for this to be done. else: ->
"""

_FIELD = re.compile(r"^(VERDICT|CHECK|EVIDENCE|ACCEPTANCE):\s*(.*)$", re.I)


def parse_verdict(text: str) -> dict | None:
    """Pull the trailing verdict block out of a turn's final text.

    Parsed from the LAST `VERDICT:` onward, not by one regex over the whole
    tail. A model that states a verdict, reconsiders, and restates would
    otherwise have the first verdict paired with the last evidence — a
    silently wrong record, which for this pipeline means a `confirmed` that
    nobody actually concluded.
    """
    from scripts.selfmod.backlog import VERDICTS

    lines = text[-6000:].splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VERDICT:"):
            start = i
    if start is None:
        return None

    fields: dict[str, list[str]] = {}
    current = None
    for line in lines[start:]:
        m = _FIELD.match(line.strip())
        if m:
            current = m.group(1).upper()
            fields[current] = [m.group(2)]
        elif current:
            fields[current].append(line)

    verdict = " ".join(fields.get("VERDICT", [])).strip().lower()
    if verdict not in VERDICTS:
        return None
    def joined(key: str, limit: int) -> str:
        return "\n".join(fields.get(key, [])).strip()[:limit]
    return {
        "verdict": verdict,
        "check": " ".join(joined("CHECK", 4000).split())[:400],
        "evidence": joined("EVIDENCE", 2000),
        "acceptance": " ".join(joined("ACCEPTANCE", 4000).split())[:600],
    }


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    new_id = queue.enqueue(
        source=NAME, kind="triage", payload={},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued backlog triage id=%d", new_id)


async def execute(item: QueueItem) -> dict[str, Any]:
    from scripts.selfmod import backlog as B, state as S
    from workers.sources._common import run_prompt_on_primary

    candidate = B.select_candidate(S.LEDGER_PATH)
    if candidate is None:
        return {"skipped": "every open backlog item has been triaged"}

    logger.info("triaging backlog #%s (%s days old): %s",
                candidate.id, candidate.age_days, candidate.name[:70])

    prompt = PROMPT.format(
        item_id=candidate.id, status=candidate.status, priority=candidate.priority,
        name=candidate.name, body=candidate.body[:6000], age=candidate.age_days,
    )
    text = await run_prompt_on_primary(prompt, max_turns=30)
    parsed = parse_verdict(text)

    if not parsed:
        # No usable verdict is itself a result: record it so the item is not
        # re-triaged forever, but do not touch its status.
        S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                        "verdict": "unverifiable", "check": "",
                        "evidence": "triage turn produced no parseable verdict block",
                        "auto": True})
        logger.warning("backlog #%s: no verdict block in the response", candidate.id)
        return {"item_id": candidate.id, "verdict": "unverifiable",
                "note": "no parseable verdict block"}

    # Retiring verdicts close the item; everything else only annotates it.
    # `confirmed` deliberately does NOT open a round — implementation goes
    # through the selfmod gate as a separate, explicit act.
    close = parsed["verdict"] in B.RETIRING
    B.record_verdict(candidate, parsed["verdict"], parsed["evidence"],
                     check=parsed["check"], close=close)

    S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                    "name": candidate.name[:200], "age_days": candidate.age_days,
                    "verdict": parsed["verdict"], "check": parsed["check"],
                    "evidence": parsed["evidence"][:1000],
                    "acceptance": parsed["acceptance"], "closed": close})

    logger.info("backlog #%s → %s%s", candidate.id, parsed["verdict"],
                " (closed)" if close else "")
    return {"item_id": candidate.id, "name": candidate.name,
            "age_days": candidate.age_days, "verdict": parsed["verdict"],
            "closed": close, "acceptance": parsed["acceptance"],
            "summary": B.summarize(S.LEDGER_PATH)}
