#!/usr/bin/env python3
"""Pairwise tool-call equivalence: did the engine's *decision* change? (#2843)

    .venvs/lloyd/bin/python eval/action_equivalence.py            # score the fixture
    .venvs/lloyd/bin/python eval/action_equivalence.py --json     # the same, machine-readable

Why this exists
---------------
Every engine-integrity verdict on this box is prose-token agreement.
`eval/engine_output_probe.py` decides per prompt on `agreement`,
`first_divergence` and two logprob deltas (`compare_prompt` at :403), tiers the
prompt `exact` or `jitter` on bitwise token agreement (:479), and every exceed
reason is an agreement or logprob threshold (:556-575); `grep -n tool_call` on
that file returns nothing. So a quant / KV-dtype / speculative-decoding change
that keeps the prose and flips *which tool* the model picks, or the path it
passes, reports clean — the model can be wrong in a way no canary can see.

This module is the missing comparator: given two parsed tool calls for the same
step, it says deterministically whether they are the same decision, a notational
variant of one, a different decision, or unreadable. It is pure — no engine, no
HTTP, no file reads — so it can be run over a recorded corpus (`--json` is what
a later job consumes) and inside a test with no engine up.

What is adjacent, and is not this
---------------------------------
* `eval/run_harness_replay_diff.py` is the one pairwise tool-call diff in the
  repo, on a *scripted* engine (:21), compared by `_sha(m.get("tool_calls"))`
  (:315). A hash has no canonicalisation, so a key-order or trailing-newline
  rewrite by a refactored harness loop reads as a decision change; and because
  the calls are fixture inputs it proves the harness loop, never the engine.
* `eval/compare_tool_choice.py` compares two runs' *aggregate* metrics against a
  derived per-metric noise floor — not pairwise, and one fixed corpus.
* `eval/cross_engine_disagreement.py` compares verdict *labels* against known
  gold on retrieval and durable-write rows, never tool-call arguments.
* `eval/tool_choice_none_probe.py` checks only whether the engine still renders
  `tools` at `tool_choice: "none"` — a wire-shape property.
* `eval/djev/action_review_calibration.py` cross-tabs 57,325 recorded worker
  calls by outcome label; it never pairs two candidate calls for one step.

The tiers, in decision precedence
---------------------------------
`protocol_invalid`  either call cannot be read as a decision: no name, or an
                    argument string that will not parse, or a parsed payload
                    that is not a JSON object. Decided FIRST: a pair with an
                    unreadable side has no decision to compare, so
                    `protocol_invalid` outranks `different_tool` even when the
                    two names differ too — reporting the name difference would
                    assert a comparison this side cannot support.
`different_tool`    both readable, names differ. Compared byte-exactly: no rule
                    here canonicalises a *name*, because picking a different
                    tool is exactly the decision this instrument exists to see.
`args_differ`       same tool, and the canonical forms differ — including when
                    the only difference is inside a value that names a target.
`canonically_equal` same tool, and the whole canonicalisation rule list makes the
                    two argument strings identical.

`args_differ` is deliberately the widest tier and the default answer to "not
notationally identical": `limit: 340` vs `limit: 180` is a different call even
though neither value names a target. Precision on `canonically_equal` is the
claim; recall on `args_differ` is the safety net.

The rule list, and the one thing it can never do
------------------------------------------------
Canonicalisation is `CANONICALISATION_RULES` — an explicit named list, applied
in order. Two rules are *structural*: they are functions of the parsed value, so
no string content can move. One rule is a *value* rule, and it is gated to
`PROSE_KEYS`. `assert_targets_preserved` then re-reads the original and the
canonical value and refuses the result if any target-naming value moved a byte,
so widening the list cannot silently widen what it may rewrite: a same-tool call
with the wrong path is a different decision, and no rule is allowed to say
otherwise. `tests/test_action_equivalence.py` proves that by installing a
hostile rule that strips trailing newlines from *every* value and watching the
guard fire — and by checking that a path differing only by a trailing newline
still lands in `args_differ`.

Exit codes (`engine_output_probe.py`'s convention)
--------------------------------------------------
0  every tier at or above MIN_PER_TIER_SCORE on the fixture
1  a tier is below it, or the comparator disagrees with a human label
2  the fixture cannot be scored at all — a tier with no labelled pair proves
   nothing about that tier, so an empty class is not a free pass
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable

#: Every tier must reach this against the fixture's human labels. 1.0 is the
#: only defensible number on a fixture this small: one wrong pair is the whole
#: margin, which is the point of shipping the labels at all.
MIN_PER_TIER_SCORE = 1.0

PROTOCOL_INVALID = "protocol_invalid"
DIFFERENT_TOOL = "different_tool"
ARGS_DIFFER = "args_differ"
CANONICALLY_EQUAL = "canonically_equal"

#: In decision precedence order — `classify` walks it top-down, and the report
#: prints it in this order so a reader sees the refusal first.
TIERS = (PROTOCOL_INVALID, DIFFERENT_TOOL, ARGS_DIFFER, CANONICALLY_EQUAL)

# ---------------------------------------------------------------------------
# What counts as "naming a target"
# ---------------------------------------------------------------------------

#: Keys whose value *is* the thing the call acts on. A wrong value here is a
#: wrong decision, whatever it looks like, so no canonicalisation rule may
#: touch it — this list is what `assert_targets_preserved` checks byte-for-byte.
#: These are real argument names off this box's tool surface (`Read.file_path`,
#: `Bash.command`, `Grep.path`, `vault_read.path`, `email_send.to`, …).
TARGET_KEYS = frozenset({
    "path", "paths", "file_path", "file", "filename", "filepath", "dir",
    "directory", "cwd", "root", "command", "cmd", "script", "url", "uri",
    "endpoint", "folderPath", "folder_path", "pattern", "glob", "query",
    "name", "entity", "symbol", "slug", "board", "agent_id", "model",
    "to", "cc", "bcc", "recipient", "address", "addressBookId", "address_book_id",
    "calendarId", "calendar_id", "taskId", "task_id", "contactId", "contact_id",
    "eventId", "event_id", "messageId", "message_id", "folderPath", "sha",
    "branch", "round_id", "session_id", "topic_id", "item_id", "file_path",
    "source_ref", "target_ref", "from_branch", "artifact_path", "source_doc",
})

#: Suffixes catch the spellings the exact list cannot enumerate (`folderPath`,
#: `task_id`, `snapshot_ts`-style ids, `*Path`). The direction is fail-closed:
#: an over-broad suffix protects a value nobody meant, and an under-broad one
#: would let a rule rewrite a target — so over-broad is the cheap mistake.
TARGET_SUFFIXES = ("_path", "_file", "_dir", "_url", "_uri", "_id", "_name",
                   "_sha", "_branch", "path", "file", "url", "uri", "command",
                   "cmd", "slug", "Id")

#: The only keys a value-level rule may touch: free prose the harness never
#: acts on, present purely so the transcript reads well. `content`/`body` are
#: NOT here — in `email_send` they are the payload, and a different payload is
#: a different decision.
PROSE_KEYS = frozenset({"summary", "note"})


def names_target(key: Any) -> bool:
    """True when a value under this key names the thing the call acts on."""
    if not isinstance(key, str):
        return False
    if key in TARGET_KEYS:
        return True
    low = key.lower()
    return any(low.endswith(s) for s in TARGET_SUFFIXES)


def _prose_keys_are_not_targets() -> None:
    """The gate on the one value-level rule, checked rather than asserted in prose."""
    bad = sorted(k for k in PROSE_KEYS if names_target(k))
    if bad:
        raise RuntimeError(
            f"PROSE_KEYS {bad} name a target, so the prose rule could rewrite a "
            "target-naming value — move the key out of PROSE_KEYS or fix names_target()")


# ---------------------------------------------------------------------------
# A parsed tool call
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """One tool call as the harness holds it: a name and the RAW argument string.

    `arguments` stays a string on purpose. Whether the two bytes differ
    *notationally* is the whole question, and `json.loads` erases that before
    you can ask it. `origin` is provenance for a report; two calls with the same
    (name, arguments) are the same decision whatever their provenance says.
    """
    name: str | None
    arguments: str | None
    origin: str = ""

    @property
    def parsed(self) -> dict | None:
        """The argument payload as a JSON object, or None if it is not one."""
        v = _load_object(self.arguments)
        return v if isinstance(v, dict) else None


def coerce(call: Any) -> ToolCall:
    """Accept a `ToolCall`, a wire dict (`{"function": {name, arguments}}`, the
    shape sessions store), a bare `{name, arguments}` dict, or a (name, args)
    pair — the four spellings a recorded corpus reaches this with."""
    if isinstance(call, ToolCall):
        return call
    origin = ""
    if isinstance(call, dict):
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        origin = str(call.get("origin", "") or fn.get("origin", ""))
        return ToolCall(name=fn.get("name"), arguments=fn.get("arguments"), origin=origin)
    if isinstance(call, (list, tuple)) and len(call) == 2:
        return ToolCall(name=call[0], arguments=call[1])
    raise TypeError(f"not a tool call: {type(call).__name__}")


def _load_object(raw: Any) -> Any:
    """Parse `raw` as JSON, or None. Never raises: an unparseable payload is a
    finding about the call, not an error in this module."""
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def protocol_reason(call: ToolCall) -> str | None:
    """Why this call cannot be decided on, or None when it can.

    The name must be a non-empty string and the payload a JSON object — the
    wire contract this engine serves (`/v1/completions` with the Hermes tool
    grammar, and `/v1/chat/completions` in OpenAI shape). An empty or absent
    argument string is refused rather than read as `{}`: a tool that takes no
    arguments still gets `"{}"` on this box, so `""` is a call that was cut off,
    which is precisely the #2617/#2320 truncation shape.
    """
    if not isinstance(call.name, str) or not call.name.strip():
        return "call has no tool name"
    if not isinstance(call.arguments, str):
        return f"argument payload is {type(call.arguments).__name__}, not a string"
    parsed = _load_object(call.arguments)
    if parsed is None:
        return "argument string is not valid JSON (a call cut off mid-object lands here)"
    if not isinstance(parsed, dict):
        return f"argument payload is a JSON {type(parsed).__name__}, not an object"
    return None


# ---------------------------------------------------------------------------
# Canonicalisation: an explicit named rule list
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonRule:
    """One canonicalisation rule, named, with the surface it may touch.

    `surface == "structure"` rules are functions of the parsed value, so no
    string content can move under them. Exactly one rule is `"value"`, and the
    guard below re-checks its work.
    """
    name: str
    surface: str
    may_touch: str
    apply: Callable[["_Canon"], "_Canon"]


@dataclass
class _Canon:
    original: dict
    value: dict
    text: str


def _sorted_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sorted_keys(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sorted_keys(v) for v in obj]
    return obj


def _strip_prose_newlines(obj: Any) -> Any:
    """Drop trailing CR/LF from a PROSE_KEYS value only — never from anything
    `names_target()`, and never from a key the rule has not been told about."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out[k] = _strip_prose_newlines(v)
            elif isinstance(v, str) and k in PROSE_KEYS and not names_target(k):
                out[k] = v.rstrip("\r\n")
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_strip_prose_newlines(v) for v in obj]
    return obj


class CanonicalisationRefused(RuntimeError):
    """A rule moved a target-naming value. The comparison stops here rather
    than reporting a changed decision as the same one."""


def _target_values(obj: Any, path: str = "$") -> dict[str, str]:
    """Every target-naming string value, keyed by its path, for the guard."""
    found: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}"
            if isinstance(v, (dict, list)):
                found.update(_target_values(v, here))
            elif names_target(k):
                found[here] = v if isinstance(v, str) else json.dumps(v, sort_keys=True)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.update(_target_values(v, f"{path}[{i}]"))
    return found


def assert_targets_preserved(original: Any, canonical: Any) -> None:
    """Refuse a canonical form in which any target-naming value moved."""
    before, after = _target_values(original), _target_values(canonical)
    moved = sorted(p for p in before if after.get(p) != before[p])
    if moved:
        raise CanonicalisationRefused(
            "canonicalisation changed a target-naming value at "
            + ", ".join(f"{p} ({before[p]!r} -> {after.get(p)!r})" for p in moved[:4]))


def _rule_key_order(c: _Canon) -> _Canon:
    return replace(c, value=_sorted_keys(c.value))


def _rule_prose_newlines(c: _Canon) -> _Canon:
    return replace(c, value=_strip_prose_newlines(c.value))


def _rule_separator_whitespace(c: _Canon) -> _Canon:
    """Serialise compactly. This is what makes whitespace around `,` and `:` and
    a trailing newline *outside* the JSON notational: the text is rebuilt from
    the parsed value, so everything between tokens goes and everything inside a
    string value survives byte-for-byte."""
    return replace(c, text=json.dumps(c.value, ensure_ascii=False, separators=(",", ":")))


#: Applied in order by `canonicalise`. The list is the contract: a rule that is
#: not named here does not run, and the guard below can veto any of them.
CANONICALISATION_RULES: tuple[CanonRule, ...] = (
    CanonRule("json_key_order", "structure",
              "object key order only; cannot reach any string value", _rule_key_order),
    CanonRule("trailing_newline_in_prose_value", "value",
              f"a trailing CR/LF on a {'/'.join(sorted(PROSE_KEYS))} value; every "
              "target-naming key is excluded by names_target()", _rule_prose_newlines),
    CanonRule("separator_whitespace", "structure",
              "whitespace between JSON tokens, leading/trailing whitespace of the "
              "payload, and a trailing newline outside it; no value content",
              _rule_separator_whitespace),
)


def canonicalise(arguments: Any) -> str:
    """The canonical text of one argument string, under the whole rule list.

    Raises `CanonicalisationRefused` if a rule moved a target-naming value, and
    `ValueError` if the payload is not a JSON object — a caller must settle
    `protocol_invalid` first (see `classify`), so a raise here is a rule bug.
    """
    parsed = _load_object(arguments)
    if not isinstance(parsed, dict):
        raise ValueError(f"canonicalise needs a JSON object payload, got {arguments!r:.80}")
    canon = _Canon(original=parsed, value=parsed, text="")
    for rule in CANONICALISATION_RULES:
        canon = rule.apply(canon)
    assert_targets_preserved(canon.original, canon.value)
    return canon.text


# ---------------------------------------------------------------------------
# The comparator
# ---------------------------------------------------------------------------

def _decide(a: ToolCall, b: ToolCall) -> tuple[str, str]:
    """(tier, why) for one pair. The single decision path `classify` and
    `explain` share, so a report cannot disagree with a verdict."""
    for side, call in (("first", a), ("second", b)):
        reason = protocol_reason(call)
        if reason:
            return PROTOCOL_INVALID, f"{side} call is unreadable: {reason}"
    if a.name != b.name:
        return DIFFERENT_TOOL, f"{a.name!r} != {b.name!r}"
    try:
        ca, cb = canonicalise(a.arguments), canonicalise(b.arguments)
    except CanonicalisationRefused as exc:
        # A rule tried to rewrite a target. That is a bug in the rule list, and
        # the honest answer is "I could not compare these", never "same".
        return PROTOCOL_INVALID, f"canonicalisation refused: {exc}"
    if ca == cb:
        return CANONICALLY_EQUAL, f"both canonicalise to {ca[:120]}"
    return ARGS_DIFFER, _first_difference(ca, cb)


def _first_difference(ca: str, cb: str) -> str:
    """A short, human-quotable account of where the two canonical forms part."""
    for i, (x, y) in enumerate(zip(ca, cb)):
        if x != y:
            return f"at byte {i}: {ca[max(0, i - 20):i + 20]!r} != {cb[max(0, i - 20):i + 20]!r}"
    return f"lengths differ: {len(ca)} vs {len(cb)}"


def classify(a: Any, b: Any) -> str:
    """The tier for one pair of parsed tool calls — `TIERS`, exactly one of them.

    Accepts `ToolCall`s or anything `coerce` reads (a wire dict, a
    `{name, arguments}` dict, a `(name, arguments)` pair).
    """
    tier, _ = _decide(coerce(a), coerce(b))
    return tier


def explain(a: Any, b: Any) -> dict[str, Any]:
    """`{"tier", "why"}` — the same decision, with the reason a corpus report needs."""
    tier, why = _decide(coerce(a), coerce(b))
    return {"tier": tier, "why": why}


# ---------------------------------------------------------------------------
# A labelled fixture of real pairs, and the score against it
# ---------------------------------------------------------------------------
#
# Every member below is a real tool call from this box's own session transcripts
# (`origin` names the session and message index), or a perturbation of one whose
# mechanism is named in the pair's `note`. No argument string here is invented:
# the tiers get precision and recall against traffic the engine actually wrote,
# and the perturbations are only ever the notations the rule list claims to
# forgive or the damage the truncation path actually does (#2617, #2320).
#
# One human label per pair, in `label`, chosen from the tier definitions at the
# top of this file — and the labels are the contract: `score_fixture` measures
# the comparator against them, not against itself.

FIXTURE: tuple[dict[str, Any], ...] = (
    {
        "id": "glob-key-order-two-sessions",
        "label": CANONICALLY_EQUAL,
        "note": "Two real calls, two sessions, the same three fields in a different "
                "order — both strings verbatim from the transcript, nothing derived.",
        "a": {"name": "Glob", "origin": "session 20261016_090238_benchmine_0517 msg7",
              "arguments": '{"summary": "Listing live bench task files", "path": "/home/alansrobotlab/obsidian/lloyd/bench", "pattern": "*.md"}'},
        "b": {"name": "Glob", "origin": "session 20261016_080439_benchmine_ea6b msg7",
              "arguments": '{"pattern": "*.md", "path": "/home/alansrobotlab/obsidian/lloyd/bench", "summary": "Listing live bench task files"}'},
    },
    {
        "id": "edit-retry-identical",
        "label": CANONICALLY_EQUAL,
        "note": "A real retry of one Edit, byte-identical (the same call stored twice in "
                "the session). A retried call is the same decision, so the tiers must not "
                "invent a difference where the engine made none.",
        "a": {"name": "Edit", "origin": "session 20261016_055852_autocode_30ba msg210",
              "arguments": '{"summary": "Importing re for the grammar assertion", "file_path": "/home/alansrobotlab/lloyd-work/SM_20261016_132019/home/lloyd/tests/test_skill_verdicts.py", "old_string": "import os\\nimport shutil", "new_string": "import os\\nimport re\\nimport shutil"}'},
        "b": {"name": "Edit", "origin": "session 20261016_055852_autocode_30ba msg216",
              "arguments": '{"summary": "Importing re for the grammar assertion", "file_path": "/home/alansrobotlab/lloyd-work/SM_20261016_132019/home/lloyd/tests/test_skill_verdicts.py", "old_string": "import os\\nimport shutil", "new_string": "import os\\nimport re\\nimport shutil"}'},
    },
    {
        "id": "bash-all-three-notations",
        "label": CANONICALLY_EQUAL,
        "note": "One real Bash call against the same call with all three notations at "
                "once: keys reordered, separators compact, a trailing newline on the "
                "summary value AND one after the closing brace. Every one of them is "
                "outside a target-naming value, so the whole rule list applies and the "
                "decision is unchanged.",
        "a": {"name": "Bash", "origin": "session 20261016_090437_autocode_4bea msg38",
              "arguments": '{"command": "cd ~/lloyd && sed -n \'240,320p\' scripts/automod/round.py", "summary": "Reading lock release path"}'},
        "b": {"name": "Bash", "origin": "derived from a: key order swapped, separators "
                                        "compacted, a \\n escape inside the summary value "
                                        "and a real newline after the closing brace",
              "arguments": '{"summary": "Reading lock release path\\n","command":"cd ~/lloyd && sed -n \'240,320p\' scripts/automod/round.py"}\n'},
    },
    {
        "id": "read-vault-note-vs-vault-read",
        "label": DIFFERENT_TOOL,
        "note": "Both real calls read a daily note. One goes through `Read` on an "
                "absolute path, the other through `vault_read` on a vault-relative one. "
                "Same intent, different tool: the tier is the tool change, not the path "
                "spelling, and `different_tool` outranks the argument question.",
        "a": {"name": "Read", "origin": "session 20261016_083009_autonomy_a7a2 msg19",
              "arguments": '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md"}'},
        "b": {"name": "vault_read", "origin": "derived from a: same note through the vault "
                                              "route, path vault-relative, as the real "
                                              "vault_read calls in these sessions spell it",
              "arguments": '{"path": "memory/2026-10-16.md", "summary": "Reading today\'s note"}'},
    },
    {
        "id": "glob-versus-read-same-directory",
        "label": DIFFERENT_TOOL,
        "note": "A real Glob of the bench directory beside a real Read of a note. Two "
                "different tools, and the payload difference is irrelevant to the tier.",
        "a": {"name": "Glob", "origin": "session 20261016_075621_benchmine_fc67 msg12",
              "arguments": '{"path": "/home/alansrobotlab/obsidian/lloyd/bench", "pattern": "*.md"}'},
        "b": {"name": "Read", "origin": "session 20261016_083009_autonomy_a7a2 msg21",
              "arguments": '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-15.md"}'},
    },
    {
        "id": "read-yesterday-not-today",
        "label": ARGS_DIFFER,
        "note": "Two real Read calls, one date apart in the filename. The prose around it "
                "is identical; the target is not. A wrong date is a wrong decision.",
        "a": {"name": "Read", "origin": "session 20261016_083009_autonomy_a7a2 msg19",
              "arguments": '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md"}'},
        "b": {"name": "Read", "origin": "session 20261016_083009_autonomy_a7a2 msg21",
              "arguments": '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-15.md"}'},
    },
    {
        "id": "grep-vault-bench-not-live-checkout",
        "label": ARGS_DIFFER,
        "note": "Two real Grep calls, same tool, different root and different pattern: one "
                "searched the live checkout, the other the vault's bench directory. Both "
                "values that differ here name a target (`path`, `pattern`), which is the "
                "kind of argument flip a dtype change could make and prose cannot see.",
        "a": {"name": "Grep", "origin": "session 20261016_075621_benchmine_fc67 msg14",
              "arguments": '{"output_mode": "files_with_matches", "path": "/home/alansrobotlab/lloyd", "pattern": "scratch-script-stdlib-shadowing"}'},
        "b": {"name": "Grep", "origin": "session 20261016_090238_benchmine_0517 msg29",
              "arguments": '{"summary": "Listing tool_called values across bench tasks", "path": "/home/alansrobotlab/obsidian/lloyd/bench", "pattern": "type: (tool_called|tool_not_called)", "-A": 1}'},
    },
    {
        "id": "read-window-differs-not-a-target",
        "label": ARGS_DIFFER,
        "note": "The same real file read with a different window: `limit: 340` beside "
                "`offset: 340, limit: 180`. Neither value names a target, and the tier is "
                "still args_differ — `args_differ` is not only about targets, and "
                "canonically_equal is not a synonym for 'close enough'.",
        "a": {"name": "Read", "origin": "session 20261016_080611_autocode_0ede msg9",
              "arguments": '{"file_path": "/home/alansrobotlab/lloyd/app/routers/sessions.py", "limit": 340}'},
        "b": {"name": "Read", "origin": "session 20261016_080611_autocode_0ede msg14",
              "arguments": '{"file_path": "/home/alansrobotlab/lloyd/app/routers/sessions.py", "offset": 340, "limit": 180}'},
    },
    {
        "id": "skill-name-only-versus-with-summary",
        "label": ARGS_DIFFER,
        "note": "Two real skills_read calls for the same skill: one sent only the name, "
                "one added a summary. Same target, different payload, so the tiers "
                "distinguish the calls rather than the intent behind them.",
        "a": {"name": "skills_read", "origin": "session 20261016_080725_autocode_1e2e msg3",
              "arguments": '{"name": "automod-change-own-code"}'},
        "b": {"name": "skills_read", "origin": "session 20261016_055852_autocode_30ba msg3",
              "arguments": '{"name": "automod-change-own-code", "summary": "Reading the automod change procedure skill"}'},
    },
    {
        "id": "read-path-with-trailing-newline",
        "label": ARGS_DIFFER,
        "note": "The clause-3 pair: a real Read against that call with one newline added "
                "INSIDE the file_path value (on the wire that is the `\\n` escape — a raw "
                "newline inside a JSON string is not JSON at all). Everything the rule list "
                "may forgive is outside a value; this newline is inside the one field that "
                "names the target, so it is a different decision and must read as one.",
        "a": {"name": "Read", "origin": "session 20261016_083009_autonomy_a7a2 msg19",
              "arguments": '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md"}'},
        "b": {"name": "Read", "origin": "derived from a: a newline appended inside file_path",
              "arguments": '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md\\n"}'},
    },
    {
        "id": "bash-command-with-trailing-newline",
        "label": ARGS_DIFFER,
        "note": "The same rule for a command-like value: the real command beside itself "
                "with a newline appended inside `command`. A command that ends in a "
                "newline is what a shell loop or a here-doc boundary changes, and on a "
                "write it is a different artifact.",
        "a": {"name": "Bash", "origin": "session 20261016_082103_autotriage_db25 msg78",
              "arguments": '{"command": "date -u \\"+%Y-%m-%d %H:%M UTC\\"", "summary": "Getting current date"}'},
        "b": {"name": "Bash", "origin": "derived from a: a newline appended inside command",
              "arguments": '{"command": "date -u \\"+%Y-%m-%d %H:%M UTC\\"\\n", "summary": "Getting current date"}'},
    },
    {
        "id": "edit-wrong-file-one-character",
        "label": ARGS_DIFFER,
        "note": "The #2320 shape on a real Edit: the same edit, the same strings, and a "
                "file_path one character shorter. A same-tool call with a wrong path is "
                "the exact miss the canary cannot see and this comparator exists for.",
        "a": {"name": "Edit", "origin": "session 20261016_080611_autocode_0ede msg181",
              "arguments": '{"file_path": "/home/alansrobotlab/lloyd-work/SM_20261016_150707/home/lloyd/tests/test_session_platform_checks.py", "new_string": "    assert sorted(reads) == sorted(f\\"{c}.json\\" for c in chats), (", "old_string": "    assert sorted(reads) == sorted(chats), ("}'},
        "b": {"name": "Edit", "origin": "derived from a: the `s` dropped from "
                                        "test_session_platform_checks.py — the wrong-path call",
              "arguments": '{"file_path": "/home/alansrobotlab/lloyd-work/SM_20261016_150707/home/lloyd/tests/test_session_platform_check.py", "new_string": "    assert sorted(reads) == sorted(f\\"{c}.json\\" for c in chats), (", "old_string": "    assert sorted(reads) == sorted(chats), ("}'},
    },
    {
        "id": "bash-args-cut-mid-object",
        "label": PROTOCOL_INVALID,
        "note": "A real Bash call cut where an engine that runs out of max_tokens cuts "
                "one: the payload ends inside the command string, so it will not parse. "
                "This is the #2617/#2320 shape the harness reports as "
                "`tool_call_truncated`, kept here as the wire leaves it.",
        "a": {"name": "Bash", "origin": "derived from session 20261016_080725_autocode_1e2e "
                                        "msg50: the first 56 bytes kept, cut inside the "
                                        "command value",
              "arguments": '{"command": "cd ~/lloyd && sed -n \'300,400p\' app/djev_sh'},
        "b": {"name": "Bash", "origin": "session 20261016_080725_autocode_1e2e msg50",
              "arguments": '{"command": "cd ~/lloyd && sed -n \'300,400p\' app/djev_shadow.py", "summary": "Reading djev_shadow record/digest"}'},
    },
    {
        "id": "empty-argument-payload",
        "label": PROTOCOL_INVALID,
        "note": "`arguments: \"\"` beside a real, complete call to the same tool. A "
                "no-argument tool still gets `{}` here (protocol_reason says why), so an "
                "empty payload is a call that never finished, not an empty decision.",
        "a": {"name": "Glob", "origin": "derived from session 20261016_080439_benchmine_ea6b "
                                        "msg4: the payload dropped entirely",
              "arguments": ""},
        "b": {"name": "Glob", "origin": "session 20261016_080439_benchmine_ea6b msg4",
              "arguments": '{"pattern": "/home/alansrobotlab/obsidian/lloyd/bench/*.md"}'},
    },
    {
        "id": "payload-is-a-json-array",
        "label": PROTOCOL_INVALID,
        "note": "The payload parses, but as a JSON array. The tool-call contract on this "
                "box is an object (`protocol_reason`), so an array is a different wire "
                "shape, and grading one as args_differ would compare fields the wire never "
                "sent — the honest answer is that there is no decision to read.",
        "a": {"name": "Glob", "origin": "derived from session 20261016_080439_benchmine_ea6b "
                                        "msg4: that call's one value lifted into a bare "
                                        "JSON array, key dropped",
              "arguments": '["/home/alansrobotlab/obsidian/lloyd/bench/*.md"]'},
        "b": {"name": "Glob", "origin": "session 20261016_080439_benchmine_ea6b msg4",
              "arguments": '{"pattern": "/home/alansrobotlab/obsidian/lloyd/bench/*.md"}'},
    },
    {
        "id": "unreadable-and-different-tool",
        "label": PROTOCOL_INVALID,
        "note": "Precedence, on real traffic: the real Read with its name removed — the "
                "shape a stream frame that never carried one arrives in — beside a real "
                "Glob. The names differ, but there is no first call to compare, so "
                "protocol_invalid outranks different_tool.",
        "a": {"name": None, "origin": "derived from session 20261016_083009_autonomy_a7a2 "
                                      "msg19: the name field removed",
              "arguments": '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md"}'},
        "b": {"name": "Glob", "origin": "session 20261016_075621_benchmine_fc67 msg12",
              "arguments": '{"path": "/home/alansrobotlab/obsidian/lloyd/bench", "pattern": "*.md"}'},
    },
)


def score_fixture(fixture: tuple[dict[str, Any], ...] = FIXTURE) -> dict[str, Any]:
    """Per-tier precision and recall of `classify` against the human labels.

    `distribution` is what the comparator said; `label_distribution` is what the
    labels said. A tier with no labelled pair is `undefined`, not 1.0 — an
    untested tier is the failure this score exists to catch, and `main` calls it
    exit 2 rather than letting an empty class look clean.
    """
    _prose_keys_are_not_targets()
    predicted: dict[str, list[str]] = {t: [] for t in TIERS}
    labelled: dict[str, list[str]] = {t: [] for t in TIERS}
    mismatches = []
    for pair in fixture:
        got = classify(pair["a"], pair["b"])
        predicted[got].append(pair["id"])
        want = pair["label"]
        if want not in labelled:
            raise ValueError(f"{pair['id']}: label {want!r} is not one of TIERS")
        labelled[want].append(pair["id"])
        if got != want:
            mismatches.append({"id": pair["id"], "labelled": want, "predicted": got,
                               "why": explain(pair["a"], pair["b"])["why"]})
    per_tier = {}
    for t in TIERS:
        n_pred, n_lab = len(predicted[t]), len(labelled[t])
        correct = len(set(predicted[t]) & set(labelled[t]))
        per_tier[t] = {
            "support": n_lab, "predicted": n_pred, "correct": correct,
            "precision": (correct / n_pred) if n_pred else None,
            "recall": (correct / n_lab) if n_lab else None,
        }
    unscored = [t for t in TIERS if per_tier[t]["support"] == 0]
    below = [t for t in TIERS if per_tier[t]["support"] and (
        (per_tier[t]["precision"] or 0.0) < MIN_PER_TIER_SCORE
        or (per_tier[t]["recall"] or 0.0) < MIN_PER_TIER_SCORE)]
    return {
        "n_pairs": len(fixture),
        "threshold": MIN_PER_TIER_SCORE,
        "rules": [r.name for r in CANONICALISATION_RULES],
        "distribution": {t: len(predicted[t]) for t in TIERS},
        "label_distribution": {t: len(labelled[t]) for t in TIERS},
        "per_tier": per_tier,
        "unscored_tiers": unscored,
        "below_threshold": below,
        "mismatches": mismatches,
    }


def render_score(score: dict[str, Any]) -> str:
    lines = [f"fixture: {score['n_pairs']} labelled pairs   "
             f"rules: {' -> '.join(score['rules'])}",
             "",
             f"{'tier':<19} {'labelled':>9} {'predicted':>10} {'precision':>10} {'recall':>8}  verdict"]
    for t in TIERS:
        s = score["per_tier"][t]
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        if t in score["unscored_tiers"]:
            verdict = "NO LABELLED PAIR — cannot be scored"
        elif t in score["below_threshold"]:
            verdict = f"BELOW {score['threshold']:g}"
        else:
            verdict = "ok"
        lines.append(f"{t:<19} {s['support']:>9} {s['predicted']:>10} "
                     f"{fmt(s['precision']):>10} {fmt(s['recall']):>8}  {verdict}")
    for m in score["mismatches"]:
        lines.append(f"  {m['id']}: labelled {m['labelled']}, comparator said "
                     f"{m['predicted']} — {m['why']}")
    if score["unscored_tiers"]:
        lines.append(f"unscored: {', '.join(score['unscored_tiers'])}")
    lines.append(f"{'PASS' if not score['below_threshold'] and not score['unscored_tiers'] else 'FAIL'}: "
                 f"{len(score['mismatches'])} of {score['n_pairs']} pairs disagree with their label")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="action_equivalence",
                                 description="Score the tool-call comparator against its labelled fixture (#2843).")
    ap.add_argument("--json", action="store_true", help="machine-readable score instead of the table")
    args = ap.parse_args(argv)
    # read FIXTURE here, not as a default argument, so a caller (or a test) that
    # rebinds the module attribute scores what it actually swapped in.
    score = score_fixture(FIXTURE)
    print(json.dumps(score, indent=1) if args.json else render_score(score))
    if score["unscored_tiers"]:
        print("exit 2: a tier with no labelled pair proves nothing about that tier",
              file=sys.stderr)
        return 2
    if score["below_threshold"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
