"""Inner Voice v4 — function-tool schemas for the observer's levers.

Replaces v3's JSON-prefill output protocol. Each lever is a function tool
the observer calls; `tool_choice="required"` forces a single tool call per
event. The tool name IS the action — no string normalization needed. Args
carry `reason` (always required, one short phrase) and `content` (required
on the levers that need text: inject, ambient, clarify).

The schemas are vLLM-format (OpenAI Chat Completions tools): a list of
dicts with `type: "function"` and a JSON Schema for `parameters`. They are
the contract between [observer.py](observer.py) and the model — keep the
descriptions terse but specific so the model doesn't need the whole vault
system prompt to disambiguate the choice.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Per-event lever tools
# ---------------------------------------------------------------------------

LEVER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": (
                "Stay silent. Use when primary is on track or mid-thought. "
                "MOST events should be noop — silence is the default. Use this "
                "when nothing is wrong."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One short phrase: why staying silent.",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inject",
            "description": (
                "Append a brief nudge to primary's chat history (will be "
                "prefixed [INNER VOICE]). Use when the primary is about to end "
                "the turn without delivering what the user asked for, or is "
                "plainly working on something else. One nudge per theme: if "
                "it did not land, say nothing more."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One short phrase: why intervening.",
                    },
                    "content": {
                        "type": "string",
                        "minLength": 5,
                        "description": (
                            "The nudge text the primary will read as a user "
                            "message. One or two sentences, direct, action-"
                            "oriented."
                        ),
                    },
                },
                "required": ["reason", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel",
            "description": (
                "Force-stop the turn. ONLY for (a) a destructive loop the "
                "primary will not break out of, or (b) a verbatim tool loop "
                "the repetition guard has already named this turn. Never for "
                "scope, drift or ignored nudges, and never as a 'task "
                "complete' lever. Anything else is downgraded to a no-op."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One short phrase: why force-stopping.",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Goal extraction tool (one-shot, turn start)
# ---------------------------------------------------------------------------

GOAL_EXTRACTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "record_goal_card",
            "description": (
                "Record the extracted goal card. Each list holds short bullet "
                "phrases (3-5 max, 8 hard cap). Empty lists if the request is "
                "conversational or has no actionable goal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "success_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                        "description": "What must be true for this turn to be done.",
                    },
                    "out_of_scope": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                        "description": "Things the agent should not pursue.",
                    },
                    "completion_signals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                        "description": "Specific outputs that indicate the goal is fully addressed.",
                    },
                },
                "required": ["success_criteria", "out_of_scope", "completion_signals"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Persistent-goal completion evaluator (post-turn, when /goal is set)
# ---------------------------------------------------------------------------

GOAL_COMPLETION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "record_goal_completion",
            "description": (
                "Judge whether the user's persistent goal has been met by the "
                "conversation so far. Be strict: only `achieved=true` when the "
                "goal's verifiable end condition is plainly satisfied by what "
                "actually happened (files written, tests passing, answer "
                "delivered). Promises of future work do NOT count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "achieved": {
                        "type": "boolean",
                        "description": (
                            "True iff the goal's end condition is plainly "
                            "satisfied. False otherwise."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 5,
                        "description": (
                            "One or two sentences. If achieved=false, this is "
                            "the user-visible follow-up prompt; be specific "
                            "about what is still missing and what the next "
                            "concrete step should be. If achieved=true, a "
                            "short note on what evidence confirmed it."
                        ),
                    },
                },
                "required": ["achieved", "reason"],
                "additionalProperties": False,
            },
        },
    },
]


# Convenience name sets for validation in observer.py.
LEVER_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in LEVER_TOOLS)
GOAL_EXTRACTION_TOOL_NAME = GOAL_EXTRACTION_TOOLS[0]["function"]["name"]
GOAL_COMPLETION_TOOL_NAME = GOAL_COMPLETION_TOOLS[0]["function"]["name"]
