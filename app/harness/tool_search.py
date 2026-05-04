"""ToolSearch — deferred tool discovery, port of Claude Agent SDK's progressive disclosure.

The harness advertises every MCP tool to vLLM on every request, which scales
poorly: tool definitions are billed as input tokens every turn, and tool-call
accuracy degrades past ~30-50 simultaneously-loaded tools. This module
implements the same trick Claude Code uses: advertise a small ``baseline``
plus a single ``ToolSearch`` meta-tool, and load full schemas on demand.

Flow:
  1. ``LoadedToolSet`` carries (catalog, baseline, loaded). ``visible_tools()``
     returns only the names in baseline ∪ loaded plus ToolSearch itself.
  2. The model sees a separate ``role: system`` reminder listing every tool's
     advertised name + one-line description (no schemas).
  3. When the model calls ``ToolSearch(query=...)``, the harness intercepts
     in ``_dispatch_one_tool_call`` (no MCP round-trip), runs ``search_tools``,
     marks the matches as loaded, and returns a ``<functions>`` block whose
     format mirrors the schema dump at the top of Claude Code's prompt.
  4. Next iteration's ``visible_tools()`` includes the newly loaded names, so
     vLLM dispatches the model's follow-up call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# ToolSearch tool definition (advertised to vLLM as a real callable tool)
# ---------------------------------------------------------------------------

TOOLSEARCH_TOOL_NAME = "ToolSearch"

TOOLSEARCH_DESCRIPTION = (
    "Fetches full schema definitions for deferred tools so they can be called.\n\n"
    "Deferred tools appear by name in <system-reminder> messages. Until fetched, "
    "only the name is known — there is no parameter schema, so the tool cannot "
    "be invoked. This tool takes a query, matches it against the deferred tool "
    "list, and returns the matched tools' complete JSONSchema definitions inside "
    "a <functions> block. Once a tool's schema appears in that result, it is "
    "callable exactly like any tool defined at the top of the prompt.\n\n"
    "Result format: each matched tool appears as one "
    "<function>{\"description\": \"...\", \"name\": \"...\", \"parameters\": {...}}</function> "
    "line inside the <functions> block — the same encoding as the tool list "
    "at the top of this prompt.\n\n"
    "Query forms:\n"
    "- \"select:Read,Edit,Grep\" — fetch these exact tools by name\n"
    "- \"notebook jupyter\" — keyword search, up to max_results best matches\n"
    "- \"+slack send\" — require \"slack\" in the name, rank by remaining terms"
)

TOOLSEARCH_OPENAI_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOLSEARCH_TOOL_NAME,
        "description": TOOLSEARCH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Query to find deferred tools. Use \"select:<tool_name>\" "
                        "for direct selection, or keywords to search."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# LoadedToolSet — per-session state object
# ---------------------------------------------------------------------------


@dataclass
class LoadedToolSet:
    """Tracks which tools the model can currently see for one session.

    ``catalog`` is the FULL OpenAI-format tool list built from the MCP pool
    discovery (post-disallowed filtering). It does NOT include ToolSearch
    itself; ToolSearch is appended in ``visible_tools()`` so it stays in
    sync with ``enabled``.

    ``baseline`` is the set of advertised names that are always visible
    (typically the seven file/shell built-ins intersected with what's
    in the catalog).

    ``loaded`` accumulates names the model has revealed via ToolSearch.
    Persists for the life of the session (see ``tool_search_cache``).

    ``enabled`` is the master switch: when False, ``visible_tools()`` returns
    the full catalog and the loaded set is irrelevant.

    ``catalog_signature`` is a stable hash of the catalog used by the cache
    to detect when a new ``run_query`` arrived with a different tool set
    (e.g. config reload changed disallowed_tools); on mismatch the cache
    drops the stale ``loaded`` set.
    """

    catalog: list[dict[str, Any]]
    baseline: set[str] = field(default_factory=set)
    loaded: set[str] = field(default_factory=set)
    enabled: bool = False
    catalog_signature: str = ""

    def visible_tools(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return list(self.catalog)
        names = self.baseline | self.loaded
        out = [t for t in self.catalog if t["function"]["name"] in names]
        out.append(TOOLSEARCH_OPENAI_TOOL)
        return out

    def is_visible(self, name: str) -> bool:
        if not self.enabled:
            return any(t["function"]["name"] == name for t in self.catalog)
        if name == TOOLSEARCH_TOOL_NAME:
            return True
        return name in self.baseline or name in self.loaded

    def mark_loaded(self, names: Iterable[str]) -> None:
        catalog_names = {t["function"]["name"] for t in self.catalog}
        for n in names:
            if n in catalog_names:
                self.loaded.add(n)


# ---------------------------------------------------------------------------
# Catalog signature — for cache invalidation
# ---------------------------------------------------------------------------


def catalog_signature(catalog: list[dict[str, Any]]) -> str:
    """Stable hash of the catalog so the cache can detect schema changes.

    We hash sorted (name, description) tuples — parameter changes within a
    tool don't invalidate the loaded set (the model already saw the schema;
    a small change won't break it). Adding/removing tools or rewording
    descriptions does invalidate, which is the conservative right call.
    """
    import hashlib

    pairs = sorted(
        (t["function"]["name"], t["function"].get("description") or "")
        for t in catalog
    )
    payload = json.dumps(pairs, separators=(",", ":")).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Catalog reminder — system message text
# ---------------------------------------------------------------------------


def format_catalog_reminder(catalog: list[dict[str, Any]]) -> str:
    """Build the ``role: system`` content block listing all deferred tools.

    Mirrors the Claude Code system-reminder shape: name on its own line,
    optional one-line description after a dash. Keeping it terse — the model
    only needs enough to know what to ask ToolSearch for.
    """
    if not catalog:
        return ""
    lines = [
        "<system-reminder>",
        "The following deferred tools are available via ToolSearch. Their "
        "schemas are NOT loaded — calling them directly will fail. Use "
        "ToolSearch with query \"select:<name>[,<name>...]\" to load tool "
        "schemas before calling them, or use keyword search to discover tools "
        "by topic:",
        "",
    ]
    for t in catalog:
        name = t["function"]["name"]
        desc = (t["function"].get("description") or "").strip()
        first_line = desc.splitlines()[0] if desc else ""
        if first_line:
            lines.append(f"- {name} — {first_line}")
        else:
            lines.append(f"- {name}")
    lines.append("</system-reminder>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# search_tools — match queries against the catalog
# ---------------------------------------------------------------------------


def search_tools(
    query: str,
    *,
    max_results: int,
    catalog: list[dict[str, Any]],
) -> tuple[list[str], str]:
    """Resolve a ToolSearch query into matched names + a ``<functions>`` block.

    Three query forms (matching Claude Code's ToolSearch):
      - ``select:Foo,Bar`` — exact name lookup
      - ``+foo bar``       — require ``foo`` substring in name; rank by ``bar``
      - ``foo bar``        — tokenized substring scoring

    Returns ``(matched_names, functions_block_text)``. Empty matches still
    return a non-empty text body — the model needs to know its query was
    understood but found nothing.
    """
    by_name = {t["function"]["name"]: t for t in catalog}
    q = (query or "").strip()

    if not q:
        # No query → return alphabetical first N as a discovery aid.
        ordered = sorted(by_name)[:max_results]
        matched = [by_name[n] for n in ordered]
        return ordered, _render_functions_block(matched, note=(
            "(empty query — showing first matches alphabetically; "
            "supply keywords or 'select:<name>' for targeted results)"
        ))

    if q.startswith("select:"):
        wanted = [n.strip() for n in q[len("select:"):].split(",") if n.strip()]
        matched_names = [n for n in wanted if n in by_name]
        missing = [n for n in wanted if n not in by_name]
        matched = [by_name[n] for n in matched_names]
        note = ""
        if missing:
            note = f"(not found: {', '.join(missing)})"
        return matched_names, _render_functions_block(matched, note=note)

    require: str | None = None
    rest = q
    if q.startswith("+"):
        # "+foo bar" — first token is required substring; rest is rank fuel.
        parts = q[1:].split(None, 1)
        require = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

    tokens = [t.lower() for t in rest.split() if t]
    scored: list[tuple[int, str]] = []
    for name, tool in by_name.items():
        name_l = name.lower()
        if require and require not in name_l:
            continue
        desc_l = (tool["function"].get("description") or "").lower()
        score = 0
        for tok in tokens:
            if tok in name_l:
                score += 3
            if tok in desc_l:
                score += 1
        if require:
            # Even with no other tokens, a required-name match is a hit.
            score += 2
        if score > 0 or (require and not tokens):
            scored.append((score, name))
    # Highest score first; tiebreak on shorter name (more specific match).
    scored.sort(key=lambda sn: (-sn[0], len(sn[1]), sn[1]))
    matched_names = [n for _, n in scored[:max_results]]
    matched = [by_name[n] for n in matched_names]
    note = ""
    if not matched_names:
        note = (
            "(no matches — try different keywords, or "
            "'select:<exact-name>' if you know the tool name)"
        )
    return matched_names, _render_functions_block(matched, note=note)


# ---------------------------------------------------------------------------
# Formatting — <functions>{...}</functions> block
# ---------------------------------------------------------------------------


def format_tool_as_function_block(openai_tool: dict[str, Any]) -> str:
    """Render one tool as a single-line ``<function>{json}</function>``.

    Mirrors the encoding Claude Code uses at the top of its prompt and in
    ToolSearch's response — picking the same format means models trained
    against that pattern can parse it without having to learn a new shape.
    """
    fn = openai_tool["function"]
    payload = {
        "description": fn.get("description") or "",
        "name": fn["name"],
        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
    }
    return f"<function>{json.dumps(payload, separators=(',', ':'))}</function>"


def _render_functions_block(tools: list[dict[str, Any]], note: str = "") -> str:
    """Wrap a list of tools in a ``<functions>...</functions>`` block."""
    lines = ["<functions>"]
    for t in tools:
        lines.append(format_tool_as_function_block(t))
    lines.append("</functions>")
    if note:
        lines.append("")
        lines.append(note)
    return "\n".join(lines)
