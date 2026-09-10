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

import logging
import re
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.autotriage")

NAME = "autotriage"
#: A long-lived re-admitter: hand-driven triages took 45-76 iterations at
#: 120k+. Held by the pool's KV gate while the primary is over budget.
LONG_LIVED = True
# `priority ASC` — lower runs sooner. Below the research/distill stream (70),
# above the implement round (40): see autocode.DEFAULT_PRIORITY.
DEFAULT_PRIORITY = 55
DEDUP_KEY = "autotriage:triage"

# How many separate items one triage may file. Measured: the first 40 runs
# filed 78, averaging 1.95 with a tail to 6. The quarantine in
# `backlog.is_quarantined` is what stops the queue feeding itself; this cap
# is about the board a human has to read afterwards. Overflow is folded into
# a single further-findings item rather than dropped, because a finding that
# lives only in EVIDENCE is still lost — see #229.
SPAWN_CAP = 3

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
2. **Decide the surface.** Which part of the system would a fix touch? \
`code` (Python under `~/lloyd`), `frontend` (`web/src`), `vault` (skills, \
SOUL.md, memories, autonomy tasks and notes under `~/obsidian`), `mixed`, or \
`external`. Hardware, robots and third-party services are `external`: the \
verdict is `not_code`, do not investigate further. **Vault content is in \
scope** — the loop lands vault changes through its own route — so an item \
about a skill, a task definition or an identity file gets a real verdict.
3. **Design a check.** A command, a file to read, a grep, a metric to query — \
something that would come out differently depending on whether the premise \
holds. Write it down before running it.
4. **Run the check.** Use Read, Grep, Glob and Bash. You are read-only **on the \
code**: do not edit, write, or commit anything under the repo, and do not start \
an automod round. The backlog is the one thing you write to — step 6 requires it.
5. **Reach a verdict** from the evidence:
   - `confirmed` — the premise still holds; the problem is real today
   - `already_done` — it was real, and something has since fixed it
   - `stale` — the premise no longer describes this system
   - `unverifiable` — the item states no claim that can be checked
   - `not_code` — not about Lloyd's own code
6. **File everything real that this item does not cover.** A `stale` verdict \
usually leaves survivors: a narrower claim that still holds, a bug you noticed \
on the way, a newer premise the old one has become. Each one becomes **its own \
backlog item, filed by you, now** — with `backlog_write_task` (board `lloyd`, no \
`task_id`, tag `spawned-by-triage`), before you write the verdict block. First \
run `backlog_tasks` to be sure no item already covers it; if one does, cite its \
number in EVIDENCE instead. Write the description as a handoff a fresh session \
can execute alone: the claim, the current state with file paths and line \
numbers, the check that shows it, and the first line \
"Split from #{item_id} during automod triage on <date>". The tool returns the \
new id; list every one under SPAWNED. **A finding that lives only in EVIDENCE \
is lost**: nobody reads this transcript for to-dos, and the item you are \
triaging is about to be closed. Filing nothing is fine when there is nothing — \
say `none` — but "those belong in two new items" with no items filed is the \
one outcome this step exists to prevent.

   **File at most {spawn_cap} separate items.** If more than {spawn_cap} real \
findings survive, file the {spawn_cap} that would change what someone does \
next, and put the remainder in **one** further item titled "Further findings \
from triage of #{item_id}", each with its own paths, line numbers and check. \
That is a cap on fan-out, not on honesty: nothing is dropped, and the \
remainder item is still a real handoff. The board is read by a human, and a \
pass that files six items per item read stops being a triage and becomes a \
second backlog.

What good triage looks like — learned from the three items closed on \
2026-09-07, each of which turned on one of these:

- **Check the item's own numbers against the live system before anything \
else.** Spring items quote entity and edge counts that are now 5-30x off. An \
item whose headline figures no longer describe the system is usually `stale` \
before you read its proposal.
- **`git log -S'<symbol>'` and `git log --oneline -- <path>`** are how you tell \
`already_done` (a commit fixed it) from `stale` (the area was rewritten or \
removed). Name the commit.
- **Ask whether the surface the item targets has any traffic.** Grep \
`sessions/*.json` for the tool it improves. Work aimed at a surface with zero \
calls is `stale` whatever its premise says.
- **If it proposes a retrieval or graph change, name the metric in \
`eval/run_eval.py` that would move.** Edge-only changes move nothing there \
(measured); such an item is `unverifiable` until a harness exists, and you \
should say which harness.
- **Check for a newer item that already covers it.** Superseded is `stale`, \
and the evidence is the newer item's number.
- **An item making several claims gets a verdict per claim.** The verdict \
block judges the item's headline premise; every claim that survives it is \
filed as its own item (step 6) and named under SPAWNED, not left in EVIDENCE.

Rules that matter:

- **`stale` and `already_done` are good outcomes.** Most of a backlog this old \
should reach them. Do not strain to confirm an item so it looks productive.
- **Never guess.** If you could not run a conclusive check, say `unverifiable` \
and explain what you would need. A wrong `confirmed` sends the automod loop \
after a problem that does not exist.
- **Quote your evidence.** File paths with line numbers, command output, commit \
SHAs. A verdict without evidence is unusable, because the point of this pass is \
that a human can audit it later.
- If you find the premise confirmed, also state **how the fix would be \
verified** — the check you just ran should fail to reproduce afterwards.
- **Some paths the loop may never touch**: `config.yaml`, `data/**`, `.env*`, \
`pytest.ini`, `.gitignore`, and under `web/` the build inputs (`package.json`, \
the lockfile, `vite.config.*`, `tsconfig*.json`). If the fix needs one of \
them the item is still `confirmed`, but a human has to land it: begin \
ACCEPTANCE with `human-only:` and name the path. The implementer skips those \
instead of spending a round finding out.

Finish with exactly this block and nothing after it:

VERDICT: <one of confirmed|already_done|stale|unverifiable|not_code>
SURFACE: <one of code|frontend|vault|mixed|external>
CHECK: <the command or method you ran, one line>
EVIDENCE: <2-4 sentences citing what you actually observed>
ACCEPTANCE: <if confirmed: what must become true for this to be done; otherwise the word none>
SPAWNED: <ids of the new items you filed in step 6, e.g. #401 #402; otherwise the word none>
"""

def _acceptance_text(value: str) -> str:
    from scripts.automod.backlog import acceptance_text
    return acceptance_text(value)


def _parse_spawned(value: str) -> list[int]:
    from scripts.automod.backlog import parse_spawned
    return parse_spawned(value)


_FIELD = re.compile(r"^(VERDICT|SURFACE|CHECK|EVIDENCE|ACCEPTANCE|SPAWNED):\s*(.*)$", re.I)


def parse_verdict(text: str, structured: dict | None = None) -> dict | None:
    """The turn's verdict, from the finalizer's object or from the text.

    Parsed from the LAST `VERDICT:` onward, not by one regex over the whole
    tail. A model that states a verdict, reconsiders, and restates would
    otherwise have the first verdict paired with the last evidence — a
    silently wrong record, which for this pipeline means a `confirmed` that
    nobody actually concluded.

    `structured` is the harness finalizer's output (a second completion under
    a JSON schema — see app/harness/finalizer.py). It wins when it carries a
    known verdict, and the text block stays in the prompt regardless: the
    finalizer is skipped whenever the turn did not end of its own accord, it
    can fail, and a verdict pipeline with no fallback would turn a transient
    engine error into a lost triage. The returned dict records which path
    produced it as `source`, so the ledger can show the fallback rate rather
    than the two being indistinguishable.
    """
    from scripts.automod.backlog import VERDICTS, SURFACES

    if isinstance(structured, dict):
        parsed = _from_structured(structured, VERDICTS, SURFACES)
        if parsed is not None:
            return parsed

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
    surface = " ".join(fields.get("SURFACE", [])).strip().lower().strip("`'\"")
    if surface not in SURFACES:
        surface = "external" if verdict == "not_code" else "code"
    def joined(key: str, limit: int) -> str:
        return "\n".join(fields.get(key, [])).strip()[:limit]
    return {
        "verdict": verdict,
        "surface": surface,
        "check": " ".join(joined("CHECK", 4000).split())[:400],
        "evidence": joined("EVIDENCE", 2000),
        # The implementer's contract. 600 cut #278's mid-way through its
        # regression guards; a contract is not the field to save bytes on.
        "acceptance": _acceptance_text(joined("ACCEPTANCE", 4000))[:3000],
        "spawned": _parse_spawned(joined("SPAWNED", 400)),
        "source": "regex",
    }


def _from_structured(obj: dict, verdicts, surfaces) -> dict | None:
    """The finalizer's object, clamped the same way the text path clamps.

    The clamps live here rather than in the schema on purpose: a `maxLength`
    is enforced by the guided decoder, so the model would stop mid-sentence
    at the limit instead of writing something shorter. Truncating afterwards
    costs a cut sentence; constraining the grammar costs the thought.
    """
    verdict = str(obj.get("verdict") or "").strip().lower()
    if verdict not in verdicts:
        return None
    surface = str(obj.get("surface") or "").strip().lower()
    if surface not in surfaces:
        surface = "external" if verdict == "not_code" else "code"
    spawned = obj.get("spawned")
    if isinstance(spawned, list):
        # Reuse the text parser per entry rather than coercing: it already
        # knows that `#401` and `401` are the same id, and a second private
        # notion of "an item id" is how the two paths would come to disagree
        # about what the model filed.
        ids: list[int] = []
        for entry in spawned:
            ids.extend(_parse_spawned(str(entry)))
        spawned = ids
    else:
        spawned = _parse_spawned(str(spawned or ""))
    return {
        "verdict": verdict,
        "surface": surface,
        "check": " ".join(str(obj.get("check") or "").split())[:400],
        "evidence": str(obj.get("evidence") or "").strip()[:2000],
        "acceptance": _acceptance_text(str(obj.get("acceptance") or ""))[:3000],
        "spawned": spawned,
        "source": "structured",
    }


DEFAULT_MAX_TURNS = 90
DEFAULT_BODY_CHARS = 30_000


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    # Budgets ride in the payload so `execute` sees the config that was live
    # when the item was queued, not whatever it is by the time it runs.
    new_id = queue.enqueue(
        source=NAME, kind="triage",
        payload={"max_turns": int(src_cfg.get("max_turns", DEFAULT_MAX_TURNS)),
                 "body_chars": int(src_cfg.get("body_chars", DEFAULT_BODY_CHARS)),
                 "structured_verdict": bool(src_cfg.get("structured_verdict", True))},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued backlog triage id=%d", new_id)


async def execute(item: QueueItem) -> dict[str, Any]:
    """Triage one item in a real session, and never mistake running out of
    room for a conclusion.

    Session-backed (see `run_prompt_in_session`) because a verdict on Lloyd's
    own code is something a human has to be able to review afterwards, and the
    direct `run_query` path leaves no transcript and attaches no observer.

    The budget check is the important part. A turn that hits `max_turns`
    stops cleanly with whatever text it had, which for a triage means no
    verdict block. That used to be recorded as `unverifiable` — a TERMINAL
    verdict — so the hardest items on the board were retired for good on first
    contact, for a reason indistinguishable from "states no checkable claim".
    Now it is recorded as `incomplete`, the item comes back, and only a second
    exhaustion retires it, with evidence that says exactly that.
    """
    from scripts.automod import backlog as B, state as S
    from workers.sources._common import DrainActive, TurnTimeout, run_prompt_in_session

    candidates, held = B.triage_pool(S.LEDGER_PATH)
    candidate = B.select_candidate(S.LEDGER_PATH)
    if candidate is None:
        # Say which empty this is. A queue drained of real work and a queue
        # holding 106 of this loop's own drafts look identical from here, and
        # only one of them means the pass is finished.
        summary = "every open backlog item has been triaged"
        if held:
            summary = (f"{summary}; {held} self-filed item(s) held until they are "
                       f"{B.SPAWN_TRIAGE_MIN_AGE_DAYS} days old")
        return {"status": "skipped", "summary": summary}
    if held:
        logger.info("triage pool: %d candidate(s), %d self-filed item(s) quarantined",
                    len(candidates), held)

    budget = int((item.payload or {}).get("max_turns") or DEFAULT_MAX_TURNS)
    body_chars = int((item.payload or {}).get("body_chars") or DEFAULT_BODY_CHARS)
    # Kill switch, in the payload like the budgets so a queued item runs under
    # the config that was live when it was enqueued. With it off the turn runs
    # identically and only the regex path reads the result — which is what
    # makes flipping it a real rollback rather than a code path nobody has
    # exercised. Items queued before this landed default to on.
    want_structured = bool((item.payload or {}).get("structured_verdict", True))
    logger.info("triaging backlog #%s (%s days old, budget %d): %s",
                candidate.id, candidate.age_days, budget, candidate.name[:70])

    prompt = PROMPT.format(
        item_id=candidate.id, status=candidate.status, priority=candidate.priority,
        name=candidate.name, body=candidate.body[:body_chars], age=candidate.age_days,
        spawn_cap=SPAWN_CAP,
    )
    try:
        run = await run_prompt_in_session(
            prompt, title=f"backlog triage #{candidate.id}: {candidate.name[:48]}",
            source=NAME, max_turns=budget, priority=1,
            final_schema=B.TRIAGE_VERDICT_SCHEMA if want_structured else None,
            final_schema_prompt=(
                "Restate the verdict block above as a single JSON object "
                "matching the schema. Same verdict, same acceptance check, "
                "same filed ids — this is a transcription, not a re-decision."
            ))
    except DrainActive as exc:
        # A landing owns the backend right now. Not a result; try next tick.
        return {"status": "skipped", "summary": f"landing in progress: {exc}"}
    except TurnTimeout as exc:
        # In-band, so the queue does not retry. Raising would send the item
        # back through the retry path, and since no verdict was recorded the
        # retry re-selects this same item — a second full triage of the item
        # that has already proved it does not finish in the time allowed.
        S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                        "verdict": B.INCOMPLETE, "reason": "turn timeout",
                        "budget": budget, "auto": True})
        logger.warning("backlog #%s: %s", candidate.id, exc)
        return {"status": "failed", "item_id": candidate.id,
                "summary": f"#{candidate.id}: {exc}"}

    session_id = run["session_id"]
    stop_reason = run.get("stop_reason")
    structured = run.get("structured") if want_structured else None
    structured_error = str(run.get("structured_error") or "")
    parsed = parse_verdict(run["text"], structured)

    if not parsed:
        if stop_reason == "max_turns":
            attempt = B.incomplete_counts(S.LEDGER_PATH).get(candidate.id, 0) + 1
            if attempt < B.MAX_INCOMPLETE_ATTEMPTS:
                S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                                "verdict": B.INCOMPLETE, "attempt": attempt,
                                "budget": budget, "session_id": session_id,
                                "num_turns": run.get("num_turns")})
                logger.warning("backlog #%s: out of budget (%d) on attempt %d — will retry",
                               candidate.id, budget, attempt)
                return {"status": "skipped", "item_id": candidate.id,
                        "verdict": B.INCOMPLETE, "attempt": attempt,
                        "session_id": session_id,
                        "summary": f"#{candidate.id} ran out of budget ({budget}); retrying later"}
            evidence = (f"ran out of iteration budget ({budget}) on {attempt} consecutive "
                        f"attempts without reaching a verdict; transcript in session "
                        f"{session_id}")
        else:
            evidence = (f"triage turn ended ({stop_reason}) with no parseable verdict "
                        f"block; transcript in session {session_id}")
        # Terminal, ledger only: the item's status is not touched, but the
        # reason is honest and the transcript is named.
        S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                        "verdict": "unverifiable", "check": "", "evidence": evidence,
                        "auto": True, "session_id": session_id,
                        "verdict_source": "none",
                        "structured_error": structured_error,
                        "stop_reason": stop_reason})
        logger.warning("backlog #%s: %s", candidate.id, evidence)
        return {"status": "success", "item_id": candidate.id, "verdict": "unverifiable",
                "session_id": session_id, "summary": evidence[:200]}

    # Retiring verdicts close the item; everything else only annotates it.
    # `confirmed` deliberately does NOT open a round — implementation is the
    # `autocode` source's job, and it is gated separately.
    close = parsed["verdict"] in B.RETIRING

    # What the model says it filed is a claim; the file on disk is the fact.
    # An id with no file behind it is recorded as unverified, never as a link.
    spawned = B.existing_ids(parsed["spawned"])
    unverified = [i for i in parsed["spawned"] if i not in spawned]
    if unverified:
        logger.warning("backlog #%s: SPAWNED names %s but no such item exists",
                       candidate.id, unverified)
    B.record_verdict(candidate, parsed["verdict"], parsed["evidence"],
                     check=parsed["check"], close=close, spawned=spawned,
                     acceptance=parsed["acceptance"])

    # The cap is a prompt instruction, and the items exist on disk by the time
    # we read SPAWNED — unfiling them would destroy real findings. So it is
    # recorded rather than enforced: a number that can be watched, on the one
    # metric that told us the pass had inverted.
    over_cap = max(0, len(spawned) - (SPAWN_CAP + 1))
    if over_cap:
        logger.warning("backlog #%s filed %d item(s) over the cap of %d(+1)",
                       candidate.id, over_cap, SPAWN_CAP)

    S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                    "name": candidate.name[:200], "age_days": candidate.age_days,
                    "spawn_cap": SPAWN_CAP, "spawned_over_cap": over_cap,
                    "verdict": parsed["verdict"], "surface": parsed["surface"],
                    "check": parsed["check"],
                    "evidence": parsed["evidence"][:1000],
                    "acceptance": parsed["acceptance"], "closed": close,
                    "spawned": spawned, "spawned_unverified": unverified,
                    # Which parser produced this verdict, and why the
                    # structured one did not when it did not. Without both,
                    # a finalizer that silently stopped working looks exactly
                    # like one that is working.
                    "verdict_source": parsed.get("source", "regex"),
                    "structured_error": structured_error,
                    "session_id": session_id, "stop_reason": stop_reason,
                    "num_turns": run.get("num_turns"), "budget": budget})

    logger.info("backlog #%s → %s%s (session %s)", candidate.id, parsed["verdict"],
                " (closed)" if close else "", session_id)
    return {"status": "success", "item_id": candidate.id, "name": candidate.name,
            "verdict": parsed["verdict"], "closed": close, "session_id": session_id,
            "summary": f"#{candidate.id} → {parsed['verdict']}"
                       f"{' (closed)' if close else ''}"}
