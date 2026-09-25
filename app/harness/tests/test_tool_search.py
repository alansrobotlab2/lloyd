"""Unit tests for app/harness/tool_search.py and tool_search_cache.py.

Covers:
  - LoadedToolSet.visible_tools / is_visible / mark_loaded
  - search_tools: select:, +token, plain keyword, empty query
  - format_catalog_reminder shape
  - format_tool_as_function_block JSON shape
  - catalog_signature stability + invalidation
  - Cache: session reuse, blank session_id is uncached, signature invalidation
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.harness import tool_search_cache
from app.harness.tool_search import (
    LoadedToolSet,
    TOOLSEARCH_OPENAI_TOOL,
    TOOLSEARCH_TOOL_NAME,
    catalog_signature,
    format_catalog_reminder,
    format_tool_as_function_block,
    search_tools,
)


def _tool(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


# ---------------------------------------------------------------------------
# LoadedToolSet
# ---------------------------------------------------------------------------


def test_visible_tools_returns_full_catalog_when_disabled():
    catalog = [_tool("Bash"), _tool("vault_search"), _tool("ambient_decide")]
    s = LoadedToolSet(catalog=catalog, baseline={"Bash"}, enabled=False)
    out = s.visible_tools()
    assert {t["function"]["name"] for t in out} == {"Bash", "vault_search", "ambient_decide"}
    assert TOOLSEARCH_TOOL_NAME not in {t["function"]["name"] for t in out}


def test_visible_tools_baseline_plus_loaded_plus_toolsearch_when_enabled():
    catalog = [_tool("Bash"), _tool("vault_search"), _tool("ambient_decide")]
    s = LoadedToolSet(catalog=catalog, baseline={"Bash"}, enabled=True)
    names = {t["function"]["name"] for t in s.visible_tools()}
    assert names == {"Bash", TOOLSEARCH_TOOL_NAME}
    s.mark_loaded(["vault_search"])
    names = {t["function"]["name"] for t in s.visible_tools()}
    assert names == {"Bash", "vault_search", TOOLSEARCH_TOOL_NAME}


def test_mark_loaded_ignores_unknown_names():
    catalog = [_tool("Bash"), _tool("vault_search")]
    s = LoadedToolSet(catalog=catalog, baseline={"Bash"}, enabled=True)
    s.mark_loaded(["nonexistent_tool", "vault_search"])
    assert s.loaded == {"vault_search"}


def test_is_visible_when_disabled_consults_full_catalog():
    catalog = [_tool("Bash"), _tool("vault_search")]
    s = LoadedToolSet(catalog=catalog, baseline=set(), enabled=False)
    assert s.is_visible("Bash")
    assert s.is_visible("vault_search")
    assert not s.is_visible("not_in_catalog")


def test_is_visible_when_enabled():
    catalog = [_tool("Bash"), _tool("vault_search")]
    s = LoadedToolSet(catalog=catalog, baseline={"Bash"}, enabled=True)
    assert s.is_visible("Bash")
    assert s.is_visible(TOOLSEARCH_TOOL_NAME)
    assert not s.is_visible("vault_search")
    s.mark_loaded(["vault_search"])
    assert s.is_visible("vault_search")


# ---------------------------------------------------------------------------
# search_tools
# ---------------------------------------------------------------------------


def test_search_select_exact_names():
    catalog = [_tool("Bash"), _tool("vault_search"), _tool("Read")]
    names, body = search_tools("select:Bash,Read", max_results=10, catalog=catalog)
    assert names == ["Bash", "Read"]
    assert "Bash" in body and "Read" in body


def test_search_select_with_missing_name_notes_it():
    catalog = [_tool("Bash")]
    names, body = search_tools("select:Bash,Nope", max_results=10, catalog=catalog)
    assert names == ["Bash"]
    assert "not found" in body
    assert "Nope" in body


def test_search_keyword_substring_in_name_outranks_description():
    catalog = [
        _tool("vault_search", "search the obsidian vault"),
        _tool("ambient_decide", "decide whether to surface a vault note"),
    ]
    names, _body = search_tools("vault", max_results=2, catalog=catalog)
    assert names[0] == "vault_search"


def test_search_required_token_form():
    catalog = [
        _tool("vault_search"),
        _tool("vault_write", "write to vault"),
        _tool("bash"),
    ]
    names, _body = search_tools("+vault write", max_results=5, catalog=catalog)
    assert "bash" not in names
    assert "vault_write" in names


def test_search_empty_query_returns_alphabetical():
    catalog = [_tool("zebra"), _tool("apple"), _tool("mango")]
    names, body = search_tools("", max_results=2, catalog=catalog)
    assert names == ["apple", "mango"]
    assert "empty query" in body


def test_search_no_match_returns_helpful_note():
    catalog = [_tool("Bash")]
    names, body = search_tools("nothing-matches-this", max_results=5, catalog=catalog)
    assert names == []
    assert "no matches" in body


def test_search_max_results_caps_output():
    catalog = [_tool(f"tool_{i}", "test") for i in range(20)]
    names, _body = search_tools("test", max_results=3, catalog=catalog)
    assert len(names) == 3


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_format_tool_as_function_block_is_single_line_json():
    block = format_tool_as_function_block(_tool("Bash", "run a shell command"))
    assert block.startswith("<function>") and block.endswith("</function>")
    inner = block[len("<function>"):-len("</function>")]
    parsed = json.loads(inner)
    assert parsed["name"] == "Bash"
    assert parsed["description"] == "run a shell command"
    assert parsed["parameters"]["type"] == "object"


def test_format_catalog_reminder_lists_names_with_descriptions():
    catalog = [_tool("Bash", "shell"), _tool("vault_search", "search vault")]
    out = format_catalog_reminder(catalog)
    assert "<system-reminder>" in out
    assert "ToolSearch" in out
    assert "select:" in out
    assert "Bash — shell" in out
    assert "vault_search — search vault" in out


def test_format_catalog_reminder_handles_empty_catalog():
    assert format_catalog_reminder([]) == ""


def test_format_catalog_reminder_handles_missing_descriptions():
    catalog = [_tool("Bash")]
    out = format_catalog_reminder(catalog)
    assert "- Bash" in out


def test_format_catalog_reminder_partitions_loaded_and_unloaded():
    """When `loaded` is supplied, the reminder splits into a 'callable now'
    section and a 'load via ToolSearch' section so the model can tell
    which tools it can already use without re-searching."""
    catalog = [
        _tool("fact_add", "add a fact"),
        _tool("vault_write", "write a note"),
        _tool("http_search", "search the web"),
    ]
    out = format_catalog_reminder(catalog, loaded={"fact_add", "vault_write"})

    # Loaded tools appear under the "callable now" header.
    assert "loaded and callable now" in out
    assert "do not re-call ToolSearch for them" in out
    assert "fact_add — add a fact" in out
    assert "vault_write — write a note" in out

    # Unloaded tools appear under the deferred header with usage hints.
    assert "available via ToolSearch" in out
    assert "select:<name>" in out
    assert "auto-loads on first use" in out
    assert "http_search — search the web" in out

    # Loaded section comes BEFORE unloaded section.
    loaded_idx = out.index("fact_add — add a fact")
    unloaded_idx = out.index("http_search — search the web")
    assert loaded_idx < unloaded_idx


def test_format_catalog_reminder_omits_loaded_section_when_none_loaded():
    """No 'callable now' header when nothing has been loaded yet — keeps the
    first-turn reminder identical to its prior look (modulo the soft-gate
    prose update)."""
    catalog = [_tool("fact_add", "add a fact")]
    out = format_catalog_reminder(catalog, loaded=set())
    assert "loaded and callable now" not in out
    assert "available via ToolSearch" in out
    assert "fact_add — add a fact" in out


def test_format_catalog_reminder_omits_unloaded_section_when_all_loaded():
    """Symmetric case: every tool already loaded → no deferred section."""
    catalog = [_tool("fact_add", "add a fact"), _tool("vault_write", "write")]
    out = format_catalog_reminder(
        catalog, loaded={"fact_add", "vault_write"},
    )
    assert "loaded and callable now" in out
    assert "available via ToolSearch" not in out


# ---------------------------------------------------------------------------
# catalog_signature
# ---------------------------------------------------------------------------


def test_catalog_signature_stable():
    catalog_a = [_tool("Bash", "x"), _tool("Read", "y")]
    catalog_b = [_tool("Read", "y"), _tool("Bash", "x")]   # reordered
    assert catalog_signature(catalog_a) == catalog_signature(catalog_b)


def test_catalog_signature_changes_when_tool_added():
    catalog_a = [_tool("Bash", "x")]
    catalog_b = [_tool("Bash", "x"), _tool("Read", "y")]
    assert catalog_signature(catalog_a) != catalog_signature(catalog_b)


def test_catalog_signature_changes_when_description_changes():
    catalog_a = [_tool("Bash", "old")]
    catalog_b = [_tool("Bash", "new")]
    assert catalog_signature(catalog_a) != catalog_signature(catalog_b)


# ---------------------------------------------------------------------------
# tool_search_cache
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


def test_cache_session_reuse_preserves_loaded_set():
    async def run():
        catalog = [_tool("Bash"), _tool("vault_search")]
        a = await tool_search_cache.get_or_create(
            "sess-1", catalog=catalog, baseline=["Bash"], enabled=True,
        )
        a.mark_loaded(["vault_search"])
        b = await tool_search_cache.get_or_create(
            "sess-1", catalog=catalog, baseline=["Bash"], enabled=True,
        )
        return a, b
    a, b = asyncio.run(run())
    assert a is b
    assert "vault_search" in b.loaded


def test_cache_invalidates_on_signature_change():
    async def run():
        cat_a = [_tool("Bash"), _tool("vault_search")]
        a = await tool_search_cache.get_or_create(
            "sess-2", catalog=cat_a, baseline=["Bash"], enabled=True,
        )
        a.mark_loaded(["vault_search"])

        cat_b = [_tool("Bash"), _tool("vault_search"), _tool("new_tool")]
        b = await tool_search_cache.get_or_create(
            "sess-2", catalog=cat_b, baseline=["Bash"], enabled=True,
        )
        return a, b
    a, b = asyncio.run(run())
    assert a is not b
    assert b.loaded == set()


def test_cache_blank_session_id_is_uncached():
    async def run():
        catalog = [_tool("Bash")]
        a = await tool_search_cache.get_or_create(
            "", catalog=catalog, baseline=["Bash"], enabled=True,
        )
        b = await tool_search_cache.get_or_create(
            "", catalog=catalog, baseline=["Bash"], enabled=True,
        )
        return a, b
    a, b = asyncio.run(run())
    assert a is not b


def test_cache_drop_forgets_session():
    async def run():
        catalog = [_tool("Bash")]
        a = await tool_search_cache.get_or_create(
            "sess-3", catalog=catalog, baseline=["Bash"], enabled=True,
        )
        a.mark_loaded(["Bash"])
        await tool_search_cache.drop("sess-3")
        b = await tool_search_cache.get_or_create(
            "sess-3", catalog=catalog, baseline=["Bash"], enabled=True,
        )
        return a, b
    a, b = asyncio.run(run())
    assert a is not b
    assert b.loaded == set()


def test_cache_refreshes_baseline_and_enabled_on_reuse():
    async def run():
        catalog = [_tool("Bash"), _tool("Read")]
        a = await tool_search_cache.get_or_create(
            "sess-4", catalog=catalog, baseline=["Bash"], enabled=True,
        )
        b = await tool_search_cache.get_or_create(
            "sess-4", catalog=catalog, baseline=["Bash", "Read"], enabled=False,
        )
        return a, b
    a, b = asyncio.run(run())
    assert a is b
    assert b.baseline == {"Bash", "Read"}
    assert b.enabled is False


# ── Catalog reminder gist (context cost) ────────────────────────────────────

def test_gist_trims_to_first_sentence():
    from app.harness.tool_search import _gist

    assert _gist("Read a file from the local filesystem. Returns content with "
                 "1-indexed line prefixes.") == "Read a file from the local filesystem."


def test_gist_keeps_short_single_sentence_whole():
    from app.harness.tool_search import _gist

    assert _gist("List kanban boards.") == "List kanban boards."


def test_gist_hard_caps_a_long_unpunctuated_line():
    from app.harness.tool_search import CATALOG_GIST_CHARS, _gist

    out = _gist("x" * 400)
    assert len(out) <= CATALOG_GIST_CHARS + 1  # +1 for the ellipsis
    assert out.endswith("…")


def test_gist_does_not_split_on_a_tiny_leading_clause():
    from app.harness.tool_search import _gist

    # "Delete a task." is under the 28-char floor, so splitting there would
    # throw away the half that says what the tool actually does.
    out = _gist("Delete a task. Archiving is reversible; deletion is not.")
    assert "Archiving" in out


def test_catalog_reminder_stays_far_below_the_full_catalog():
    """The reminder is a recognition aid, not a second copy of the schemas."""
    import json

    from app.harness.tool_search import format_catalog_reminder

    catalog = [
        {"type": "function", "function": {
            "name": f"tool_{i}",
            "description": "A tool that does a thing. " + "Detail sentence. " * 20,
            "parameters": {"type": "object", "properties": {
                "a": {"type": "string", "description": "x" * 200}}},
        }}
        for i in range(100)
    ]
    reminder = format_catalog_reminder(catalog, loaded=set())
    assert len(reminder) < len(json.dumps(catalog)) / 4


# ── Catalog gist fidelity (#935) ─────────────────────────────────────────────
#
# `_gist` keeps the first sentence of a description and hard-caps it at
# CATALOG_GIST_CHARS, so a tool whose "when to use / when NOT to use" clause
# sits in its second sentence loses it from the catalog reminder. The reminder
# is the only place the model sees an unloaded tool at decision time once
# `harness.tool_search.enabled` is on, and that flag is off today (#456), so
# this is a re-enable hazard: flip the boolean and the truncation resumes with a
# green suite. This rail runs the item's AST scan over the real descriptions
# and holds every loss to an explicit, reasoned exception. `_gist` and its
# budget are deliberately not touched here — #639 owns the token cost.

_COND = r"\b(Prefer|instead of|never|do not|WHENEVER|before)\b"

#: Tools whose description carries conditional guidance the gist drops today,
#: each with why the loss is tolerable. Adding a name here is a decision that
#: the tool's first sentence is enough to pick it from the catalog; the fix for
#: a NEW entry is usually to move the trigger clause into the first sentence.
GIST_LOSS_EXCEPTIONS: dict[str, str] = {
    "automod_amend_clause": "the 'never weaker' rule is for the argument, not for choosing the tool",
    "autonomy_config": "'never writes' is a read-only note; the name and gist already say read",
    "autonomy_health": "'read the window before trusting' is about interpreting the result",
    "backlog_boards": "'before filtering with backlog_tasks' is ordering advice; gist says what it lists",
    "browser_snapshot": "'never reads as complete' describes the output's honesty, not when to call",
    "browser_evaluate": "'inside that frame instead of the top' is a parameter detail",
    "Task": "the resume-instead-of-restart rule needs the task_id, which the schema carries",
    "graph_explain": "'use before hand-searching' is ordering advice; the gist names the graph",
    "graph_affected": "'call this before changing a shared function' is ordering advice",
    "http_fetch": "the first sentence already routes: page here, API to http_request, localhost to Bash",
    "research_propose": "'instead of researching it twice' describes the dedupe list it returns",
    "research_next": "'never takes work away' is a read-only note; 'peek' already says so",
    "research_complete": "'never rewritten' is a write-once note about an outcome already recorded",
    "research_list": "'before proposing anything new' is ordering advice; the gist names the filters",
    "research_stats": "'start here before' is ordering advice; the gist names the counts",
    "vault_overview": "'orient before searching' is ordering advice; the gist says summarize",
}


def _gist_losses() -> dict[str, tuple[int, int]]:
    """{tool name: (description chars, gist chars)} for every literal
    `Tool(description=…)` in agent_mcp/*.py whose description carries a
    conditional phrase that its gist does not."""
    import ast
    import re
    from pathlib import Path

    from app.harness.tool_search import CATALOG_GIST_CHARS, _gist

    root = Path(__file__).resolve().parents[3]
    lost: dict[str, tuple[int, int]] = {}
    for path in sorted((root / "agent_mcp").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "Tool"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            desc, name = kw.get("description"), kw.get("name")
            if not (isinstance(desc, ast.Constant) and isinstance(desc.value, str)):
                continue
            if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                continue
            if len(desc.value) <= CATALOG_GIST_CHARS:
                continue
            gist = _gist(desc.value)
            if re.search(_COND, desc.value) and not re.search(_COND, gist):
                lost[name.value] = (len(desc.value), len(gist))
    return lost


def test_gist_keeps_conditional_guidance_or_is_excepted():
    lost = _gist_losses()
    assert lost, "the scan found no literal Tool(description=…) at all; it is broken"
    unexpected = {n: lost[n] for n in lost if n not in GIST_LOSS_EXCEPTIONS}
    assert not unexpected, (
        "these tools' catalog gists drop their conditional guidance and are not "
        "in GIST_LOSS_EXCEPTIONS; move the trigger clause into the first "
        f"sentence, or except them with a reason: {unexpected}")
    # The list may not rot: a name that stopped losing has to come off, or the
    # exceptions stop describing the tree.
    stale = sorted(n for n in GIST_LOSS_EXCEPTIONS if n not in lost)
    assert not stale, f"no longer losing guidance, remove from GIST_LOSS_EXCEPTIONS: {stale}"
    assert all(GIST_LOSS_EXCEPTIONS[n].strip() for n in GIST_LOSS_EXCEPTIONS)
