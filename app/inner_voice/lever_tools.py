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
                "prefixed [INNER VOICE]). Use when primary is drifting from "
                "the goal, looping, or about to terminate without answering. "
                "After two injects on the same theme, escalate to cancel "
                "instead of injecting again."
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
                "Force-stop the turn. Reserve for: (a) primary in a destructive "
                "loop it won't break out of; (b) primary has ignored 2+ injects "
                "on the same theme; (c) tight loop calling the same tool with "
                "same args. Do NOT use cancel as a 'task complete' lever — the "
                "harness terminates naturally on text-only iterations. Do NOT "
                "cancel mid-tool-sequence."
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
    {
        "type": "function",
        "function": {
            "name": "ambient",
            "description": (
                "Queue a follow-up turn that fires after this one finishes. "
                "Use when something is worth surfacing but doesn't need to "
                "interrupt the current turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One short phrase: why queueing follow-up.",
                    },
                    "content": {
                        "type": "string",
                        "minLength": 5,
                        "description": "Body of the ambient follow-up message.",
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
            "name": "clarify",
            "description": (
                "Ask the user a question; pauses primary until they reply. Use "
                "ONLY when the goal is genuinely ambiguous in a way that will "
                "materially change the answer. Sparingly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One short phrase: why ambiguity needs user input.",
                    },
                    "content": {
                        "type": "string",
                        "minLength": 5,
                        "description": "The clarifying question to surface to the user.",
                    },
                },
                "required": ["reason", "content"],
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


# Convenience name sets for validation in observer.py.
LEVER_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in LEVER_TOOLS)
GOAL_EXTRACTION_TOOL_NAME = GOAL_EXTRACTION_TOOLS[0]["function"]["name"]
