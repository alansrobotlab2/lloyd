#!/usr/bin/env python3
"""
Lloyd MCP Server: Subliminal — vault context injection.

In Lloyd, subliminal context injection is handled differently than in Hermes.
Instead of a pre_llm_call hook, this MCP server provides a tool that the
prompt_builder can call to get vault context for a query.

The SDK's system_prompt parameter handles injection — this server just
provides the retrieval mechanism.

Tools: subliminal_recall
"""

import json
import re
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent

LLOYD_HOME = Path.home() / "lloyd"
SUBLIMINAL_PATH = Path.home() / "obsidian" / "lloyd" / "SOUL.md"

_NOISE_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "his", "her",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "between",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "just", "also", "very", "really", "quite", "too", "much",
    "ok", "okay", "yeah", "yes", "nah", "sure", "right",
    "lets", "let", "go", "going", "get", "got", "getting",
    "want", "wants", "wanted", "know", "knows", "knew",
    "think", "thinks", "thought", "look", "looking", "looked",
    "take", "takes", "took", "make", "makes", "made",
    "now", "some", "any", "all", "each", "every", "both",
    "up", "out", "over", "down", "off", "away",
    "here", "there", "thing", "things", "stuff",
    "left", "done", "next", "back", "ready", "still", "already",
    "tell", "show", "give", "put", "run", "running", "ran",
    "come", "came", "see", "saw", "seen", "say", "said",
    "try", "tried", "use", "used", "using", "work", "working", "worked",
    "start", "started", "stop", "stopped", "keep", "kept",
    "set", "check", "handle", "update", "updated",
}

app = Server("lloyd-subliminal")


def _extract_keywords(text: str) -> str:
    words = re.sub(r"[^a-z0-9\s\-_\.]", " ", text.lower()).split()
    keywords = list(dict.fromkeys(w for w in words if len(w) > 1 and w not in _NOISE_WORDS))
    return " ".join(keywords)


def _load_soul() -> str | None:
    if not SUBLIMINAL_PATH.exists():
        return None
    try:
        content = SUBLIMINAL_PATH.read_text(encoding="utf-8").strip()
        if content.startswith("---"):
            end = content.find("\n---\n", 3)
            if end != -1:
                content = content[end + 5:].strip()
        return content or None
    except Exception:
        return None


@app.list_tools()
async def list_tools():
    return [
        Tool(name="subliminal_recall", description="Extract keywords from a query and return the operating contract + keyword string for vault recall.", inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User message to extract keywords from"},
            },
            "required": ["query"],
        }),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "subliminal_recall":
        query = arguments.get("query", "").strip()
        if not query or len(query) < 10:
            return [TextContent(type="text", text=json.dumps({"skipped": True}))]

        keywords = _extract_keywords(query)
        soul = _load_soul()

        result = {"keywords": keywords}
        if soul:
            result["operating_contract"] = soul

        return [TextContent(type="text", text=json.dumps(result))]

    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

