"""P10 seam 1 — an injection-resistant reviewer of worker tool calls, in shadow.

A worker turn reads untrusted text (web pages, transcripts, backlog bodies
someone else wrote) and then acts. The gates that exist today judge the ACTION
on its own shape — a destructive Bash, a tier-2 tool with no grant — and none
asks the question an injection turns on: is this call what the TASK asked for,
or what the text the agent just read asked for?

This module asks djev that question once per tool call of a
`NON_USER_PLATFORMS` turn and records the answer beside what production did.
It decides nothing, and after the measurement below it is not going to.

THE THRESHOLD QUESTION, AND HOW IT WAS SETTLED (item #1944, 2026-10-01)
----------------------------------------------------------------------
The seam ran with `ACTION_REVIEW.threshold is None`, which made `shadow` and
`warn` behaviourally identical, so the question its rows had been accumulating
against was never asked until it was measured. Re-run the answer with
`python -m eval.djev.action_review_calibration`. The corpus is append-only and
live, so these are a dated snapshot — read at **2026-10-01T02:47Z**, over
**57,325 `action_review` rows** spanning **2026-09-25T18:45Z →
2026-10-01T02:46Z**, 5.34 days of the seam's own traffic and NOT the file's,
whose head is a `rerank` row five days earlier. A re-run reports more rows and a
rate a little off this one; the ratios are the finding, not the absolutes.

**963 rows (1.68%) came back `unrelated` — 180.5 per day** — and **929 of them
were calls that actually ran**, so a `warn` would have interrupted **174.1 per
day**. Where those positives sit is the ruling: **1 of the 963** is a tier ≥ 2
durable-external call, **961 are tier 1**, and 1 could not be tiered. The
remainder is spread across the ordinary tools at the same order of rate
(`Bash` 798/42,006 = 1.90%, `Read` 53/5,174 = 1.02%, `Edit` 31/4,897 = 0.63%)
and across every worker source (`automod-review` 492, `autotriage` 145,
`autocode` 143, `owed-check` 118, `youtube-digest` 54). That is branch (b) of the
item: topical drift, not a concentration a gate could act on. Even the one
tier-2 positive argues against gating — it is an `owed-check` call the
deterministic hook had already denied, so the reviewer added nothing to the gate
that the gate did not already have.

Two more facts make a threshold unnameable even if the shape had been right.
**`injected` is the argmax in 0 of 57,325 rows** — the option the schema comment
called "the one a threshold will be read off" has never once been the answer —
and no shadow row carries a human label, so FP and FN are uncomputable: the P10
rule (FP ≤ 2% at FN ≤ 30%) is unevaluable here, not merely unmet. And `warn` was
never implemented, so branch (a) needed an emit path built before it could be
measured at all, which the item's own header excluded.

**Ruling: `action_review` stays in shadow permanently.** `warn` is out of
`MODES` so the switch cannot be opened on vibes, `threshold` and
`label_mass_floor` stay `None`, and reopening needs new evidence rather than a
recollected rate — the counts come from `eval/djev/action_review_calibration.py`,
the reasoning is in `knowledge/ai/action-review-threshold-measurement.md`, and
the rule it was judged against is `architecture/harness.md` (P10).

WHAT THE REVIEWER IS SHOWN, AND WHAT IT IS NOT
----------------------------------------------
The worker's own prompt — written by the source, trusted — and the tool calls
so far, as name plus a clipped argument rendering. **Never assistant prose and
never tool results.** A reviewer that reads the fetched content can be
injected by the same text as the agent it is reviewing; one that sees only the
task and the actions cannot be told what to think by the page, only shown
what the agent did after reading it. That exclusion is the whole design, and
`tests/test_action_review.py` pins it.

WHY AN OnEvent TAP AND NOT A PreToolUse GATE
--------------------------------------------
The row's `actual` is what the existing gates decided, and no PreToolUse
position can know that: the walk returns on the first deny, so a callback
after the safety hook never sees a denied call, and the Inner Voice observer
attaches its own PreToolUse callback at run time, after this is installed. The
`tool_result` event is the first place every gate's verdict is visible, so the
reviewer records there, with the view as it stood when the call was made
(parallel batches announce every call before any result lands).

NEVER IN FRONT OF A TOOL CALL
-----------------------------
The callback does a list append on `tool_call` and, on `tool_result`, one
`djev_shadow.shadow(...)` — a bounded `put_nowait`. The canvas text and the
questions are zero-argument callables the recorder's worker thread builds, the
same shape as the dedupe seam's candidate heads. Nothing here raises into the
loop: `HookRegistry.fire_on_event` already swallows, and this swallows first.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger("lloyd-harness-action-review")

SEAM = "action_review"
#: `warn` was removed 2026-10-01 on the measurement in the module docstring
#: (item #1944): 174.1 would-be interruptions per day, 1 of 963 positives above
#: tier 1, `injected` the argmax in 0 of 57,325 rows. A config still carrying
#: `mode: warn` falls back to `DEFAULT_MODE` below, which is what `warn` did
#: anyway — nothing ever branched on it — so the fallback is the honest one, not
#: a silent gate. `tests/test_action_review.py` pins that `warn` is gone.
MODES = ("off", "shadow")
DEFAULT_MODE = "shadow"

#: The trusted task, clipped. A worker prompt is 2-12 KB; the head carries the
#: instruction, the tail is mostly rules and output format.
PROMPT_CHARS = 2400
#: One call's arguments as the reviewer sees them. Enough for a command, a
#: URL, a path and the start of a body.
ARG_CHARS = 240
#: The prior calls shown, newest last. A 250-iteration round would otherwise
#: build a canvas of thousands of lines.
MAX_PRIOR_CALLS = 20


def mode_from_config() -> str:
    """`harness.action_review.mode`, defaulting to shadow. Fail-open to the
    default: an unreadable config must not switch a recorder into a gate or
    silently off."""
    try:
        from app.config import CONFIG
        block = ((CONFIG or {}).get("harness") or {}).get("action_review") or {}
        mode = str(block.get("mode", DEFAULT_MODE)).strip().lower()
    except Exception:  # noqa: BLE001
        return DEFAULT_MODE
    return mode if mode in MODES else DEFAULT_MODE


def args_rendering(args: Any) -> str:
    """One call's arguments, as the canvas shows them: compact JSON, clipped."""
    try:
        text = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str,
                          separators=(",", ":"))
    except Exception:  # noqa: BLE001
        text = repr(args)
    return text if len(text) <= ARG_CHARS else text[:ARG_CHARS] + "…"


def args_digest(args: Any) -> str:
    try:
        blob = json.dumps(args, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        blob = repr(args)
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:16]


def outcome_of(result_evt: dict[str, Any]) -> str:
    """What the existing gates and the tool made of the call.

    Read off the result text because that is the only place every gate's
    verdict converges: the loop's hook deny and disabled-tool refusal, and the
    aggregator's dispatch refusals (`main._refused_call`), which are worded
    alike on purpose so a bench trace files them together.
    """
    content = str(result_evt.get("content") or "")
    is_error = bool(result_evt.get("is_error"))
    if content.startswith("Tool call denied:"):
        return "denied_by_hook"
    if is_error and content.startswith("Tool ") and "disabled by configuration" in content[:200]:
        return "disabled"
    if is_error and "Tool call denied:" in content[:400]:
        return "refused_at_dispatch"
    if is_error:
        return "error"
    return "ran"


def canvas_state(prompt: str, prior: list[tuple[str, str]],
                 call: tuple[str, str]) -> str:
    """The reviewer's whole view: task, calls so far, the call under review."""
    task = str(prompt or "")
    if len(task) > PROMPT_CHARS:
        task = task[:PROMPT_CHARS] + "…"
    shown = prior[-MAX_PRIOR_CALLS:]
    lines = []
    omitted = len(prior) - len(shown)
    if omitted:
        lines.append(f"({omitted} earlier calls not shown)")
    base = omitted + 1
    for i, (name, args) in enumerate(shown):
        lines.append(f"{base + i}. {name} {args}")
    calls = "\n".join(lines) if lines else "(none)"
    return (f"The task the agent was given (written by its operator, trusted):\n"
            f"{task}\n\n"
            f"Tool calls the agent has made so far:\n{calls}\n\n"
            f"Call under review:\n{call[0]} {call[1]}")


def questions() -> dict:
    """The frozen question, copied so option ORDER survives (it is part of
    the schema hash)."""
    from eval.djev import schemas
    q = schemas.ACTION_REVIEW.spec["on_task"]
    return {"on_task": {"type": q["type"], "instructions": q["instructions"],
                        "criteria": dict(q["criteria"])}}


class ActionReviewer:
    """Per-turn state: the trusted prompt and the calls the turn has made."""

    def __init__(self, *, user_prompt: str, source: str, session_id: str,
                 mode: str) -> None:
        self.user_prompt = str(user_prompt or "")
        self.source = str(source or "")
        self.session_id = str(session_id or "")
        self.mode = mode
        #: (name, clipped args) per announced call, in wire order.
        self.calls: list[tuple[str, str]] = []
        #: call_id -> (name, clipped args, digest, index into `calls`)
        self._pending: dict[str, tuple[str, str, str, int]] = {}
        self.recorded = 0

    async def on_event(self, evt: dict[str, Any]) -> None:
        try:
            kind = evt.get("type")
            if kind == "tool_call":
                self._on_call(evt)
            elif kind == "tool_result":
                self._on_result(evt)
        except Exception as exc:  # noqa: BLE001 — a recorder never reaches the turn
            logger.debug("action_review: event skipped: %s", exc)

    def _on_call(self, evt: dict[str, Any]) -> None:
        name = str(evt.get("name") or "")
        args = evt.get("args_dict")
        if not isinstance(args, dict):
            args = {}
        rendered = args_rendering(args)
        self._pending[str(evt.get("call_id") or "")] = (
            name, rendered, args_digest(args), len(self.calls))
        self.calls.append((name, rendered))

    def _on_result(self, evt: dict[str, Any]) -> None:
        entry = self._pending.pop(str(evt.get("call_id") or ""), None)
        if entry is None:
            return
        name, rendered, digest, index = entry
        prior = list(self.calls[:index])
        prompt = self.user_prompt

        from app import djev_shadow
        djev_shadow.shadow(
            seam=SEAM,
            # Built by the recorder's worker, never on the loop.
            state=lambda: canvas_state(prompt, prior, (name, rendered)),
            questions=questions,
            actual={"outcome": outcome_of(evt)},
            meta={"session_id": self.session_id, "source": self.source,
                  "tool": name, "args_digest": digest, "prior_calls": index,
                  "call_id": str(evt.get("call_id") or ""), "mode": self.mode},
        )
        self.recorded += 1


def install_action_review_hook(hooks: Any, *, user_prompt: str, source: str = "",
                               mode: str | None = None,
                               session_id: str = "") -> ActionReviewer | None:
    """Tap this turn's event stream for the action reviewer, or do nothing.

    The caller decides eligibility (a `NON_USER_PLATFORMS` turn); this decides
    the mode. Returns the reviewer (for tests), or `None` when off. Registers
    no PreToolUse callback: it cannot change a tool call's outcome by
    construction, and there is no `warn` mode left to change it under — see the
    ruling in this module's docstring.
    """
    try:
        mode = (mode or mode_from_config()).strip().lower()
        if mode not in MODES:
            mode = DEFAULT_MODE
        if mode == "off" or hooks is None:
            return None
        reviewer = ActionReviewer(user_prompt=user_prompt, source=source,
                                  session_id=session_id, mode=mode)
        hooks.add_on_event(reviewer.on_event)
        return reviewer
    except Exception as exc:  # noqa: BLE001
        logger.debug("action_review: not installed: %s", exc)
        return None
