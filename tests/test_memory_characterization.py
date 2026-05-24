"""Characterization tests for the agent_mcp knowledge-graph helpers.

Originally written against the monolithic agent_mcp/memory.py to pin
current behavior before PR 1's extraction. After PR 5 the module was
split into facts/vault/session/_shared, with a `memory.py` shim
re-exporting everything for backward compat. The cleanup PR (post-PR-4)
deletes the shim, so this file imports directly from the canonical
modules.

Run: .venvs/lloyd/bin/python -m pytest tests/test_memory_characterization.py

Scope: pure functions and the public tool-list shape. I/O-bound tests
(disk-touching _resolve_entity round-trips) are deferred to a later PR
because they require patching FACTS_ROOT / ALIASES_PATH across modules,
which is easier after PR 3 makes path resolution explicit.

The `memory` namespace alias below is just to keep these tests readable
— `memory._token_overlap` reads better than juggling four module names
across many short tests. All names below resolve to `_shared.py`.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# `_shared` holds all the pure helpers and stopword sets exercised by
# this file. Aliasing keeps the assertion expressions concise.
from agent_mcp import _shared as memory  # noqa: E402

# For test_list_tools_*: aggregate the 20 tools across the three split
# modules. After the shim is gone there's no `memory.list_tools()`.
from agent_mcp import facts as _facts_mod  # noqa: E402
from agent_mcp import vault as _vault_mod  # noqa: E402
from agent_mcp import session as _session_mod  # noqa: E402


async def _aggregated_list_tools() -> list:
    """Aggregate tools across facts/vault/session — replaces the
    deleted `agent_mcp.memory.list_tools()` from PR 5's shim."""
    out: list = []
    for mod in (_facts_mod, _vault_mod, _session_mod):
        out.extend(await mod.list_tools())
    return out


# ---------------------------------------------------------------------------
# Pure helpers — _token_overlap
# ---------------------------------------------------------------------------

def test_token_overlap_jaccard():
    # "foo bar" vs "bar baz": intersection {bar}, union {foo, bar, baz} → 1/3
    result = memory._token_overlap("foo bar", "bar baz")
    assert abs(result - 1.0 / 3.0) < 1e-9, f"expected 1/3, got {result}"


def test_token_overlap_empty():
    assert memory._token_overlap("", "") == 0.0
    assert memory._token_overlap("foo", "") == 0.0
    assert memory._token_overlap("", "foo") == 0.0


def test_token_overlap_identical():
    assert memory._token_overlap("foo bar", "foo bar") == 1.0


def test_token_overlap_case_insensitive():
    # _token_overlap lowercases internally
    assert memory._token_overlap("Foo BAR", "foo bar") == 1.0


# ---------------------------------------------------------------------------
# Pure helpers — _levenshtein
# ---------------------------------------------------------------------------

def test_levenshtein_classic():
    assert memory._levenshtein("kitten", "sitting") == 3


def test_levenshtein_empty():
    assert memory._levenshtein("", "abc") == 3
    assert memory._levenshtein("abc", "") == 3
    assert memory._levenshtein("", "") == 0


def test_levenshtein_identical():
    assert memory._levenshtein("hello", "hello") == 0


def test_levenshtein_swap_order_symmetric():
    # The function reorders by length, so verify symmetry
    assert memory._levenshtein("abc", "abcd") == memory._levenshtein("abcd", "abc")


# ---------------------------------------------------------------------------
# Pure helpers — _fuzzy_entity_match
# ---------------------------------------------------------------------------

def test_fuzzy_entity_match_exact_case_insensitive():
    # exact match short-circuits
    assert memory._fuzzy_entity_match("Foo", ["foo", "bar"]) == "foo"
    assert memory._fuzzy_entity_match("FOO", ["foo"]) == "foo"


def test_fuzzy_entity_match_below_threshold():
    # totally different → None
    assert memory._fuzzy_entity_match("xyz", ["abc", "def"]) is None


def test_fuzzy_entity_match_length_ratio_block():
    # "agent prompt constraint" should NOT match "agent" (len ratio > 2).
    # This is the #310 Tier 4 fix — short canonical names can't swallow long
    # specific names anymore.
    assert memory._fuzzy_entity_match("agent prompt constraint", ["agent"]) is None


def test_fuzzy_entity_match_typo_too_strict():
    # Single-char transposition with NO shared tokens does NOT pass the 0.85
    # threshold. token_overlap=0 (different unique words), lev_score=0.6,
    # combined=0.36. This is intentional per #310 — bumped from 0.7 → 0.85 to
    # stop short canonicals from swallowing typos.
    assert memory._fuzzy_entity_match("Lloyd", ["Llyod"]) is None


def test_fuzzy_entity_match_separator_variation():
    # Multi-token names where only the separator differs DO match.
    # "lloyd-mc" vs "lloyd mc": token_overlap=1.0, lev_score=0.875,
    # combined ≈ 0.925 > 0.85.
    result = memory._fuzzy_entity_match("Lloyd-MC", ["Lloyd MC"])
    assert result == "Lloyd MC", f"expected 'Lloyd MC', got {result!r}"


def test_fuzzy_entity_match_empty_candidates():
    assert memory._fuzzy_entity_match("anything", []) is None


# ---------------------------------------------------------------------------
# Pure helpers — _parse_fact_frontmatter / _write_fact_frontmatter
# ---------------------------------------------------------------------------

def test_parse_fact_frontmatter_no_frontmatter():
    assert memory._parse_fact_frontmatter("just body, no front matter") == {}


def test_parse_fact_frontmatter_unclosed():
    # Missing closing --- → return {}
    assert memory._parse_fact_frontmatter("---\nfoo: bar\nno close") == {}


def test_parse_fact_frontmatter_empty_block():
    # `---\n---\n` — empty body, parses to {} (yaml.safe_load("") returns None → {})
    assert memory._parse_fact_frontmatter("---\n---\nbody") == {}


def test_parse_fact_frontmatter_round_trip():
    data = {
        "type": "facts",
        "entity": "Lloyd",
        "category": "general",
        "facts": [{"id": "gene-abcd", "fact": "test fact", "confidence": 0.9}],
    }
    yaml_text = memory._write_fact_frontmatter(data)
    # _write_fact_frontmatter emits "---\n<yaml>---\n" with NO closing newline
    # before the second ---. Parsing that + a body should round-trip.
    parsed = memory._parse_fact_frontmatter(yaml_text + "\n# body here\n")
    assert parsed == data, f"round-trip mismatch:\n{parsed!r}\nvs\n{data!r}"


def test_write_fact_frontmatter_starts_with_marker():
    out = memory._write_fact_frontmatter({"k": "v"})
    assert out.startswith("---\n"), f"expected leading ---, got {out!r}"
    assert out.endswith("---\n"), f"expected trailing ---\\n, got {out!r}"


# ---------------------------------------------------------------------------
# Stopword sets — pin current membership
# ---------------------------------------------------------------------------

def test_entity_stopwords_basic_membership():
    # If these change, downstream entity extraction shifts. Pin them.
    must_contain = {"the", "a", "an", "and", "or", "is", "are", "this", "that"}
    assert must_contain.issubset(memory._ENTITY_STOPWORDS), (
        f"missing entity stopwords: {must_contain - memory._ENTITY_STOPWORDS}"
    )
    # Sanity: should NOT contain content words
    must_not_contain = {"agent", "lloyd", "config", "task"}
    assert not (must_not_contain & memory._ENTITY_STOPWORDS), (
        f"content words leaked into entity stopwords: "
        f"{must_not_contain & memory._ENTITY_STOPWORDS}"
    )


def test_scoring_stopwords_superset_of_entity():
    # _SCORING_STOPWORDS = _ENTITY_STOPWORDS | {"task", "backlog", "item", ...}
    assert memory._ENTITY_STOPWORDS.issubset(memory._SCORING_STOPWORDS)
    assert "task" in memory._SCORING_STOPWORDS
    assert "backlog" in memory._SCORING_STOPWORDS


def test_query_stopwords_superset_of_entity():
    # _QUERY_STOPWORDS = _ENTITY_STOPWORDS | {pronouns, framing verbs, ...}
    assert memory._ENTITY_STOPWORDS.issubset(memory._QUERY_STOPWORDS)
    assert "walk" in memory._QUERY_STOPWORDS
    assert "tell" in memory._QUERY_STOPWORDS
    assert "happens" in memory._QUERY_STOPWORDS


# ---------------------------------------------------------------------------
# Path constants — sanity
# ---------------------------------------------------------------------------

def test_path_constants_resolved_under_home():
    home = Path.home()
    assert memory.FACTS_ROOT == home / "obsidian" / "facts"
    assert memory.ALIASES_PATH == memory.FACTS_ROOT / "entity-aliases.json"
    assert memory.VAULT == home / "obsidian"


# ---------------------------------------------------------------------------
# Tool registration — pin the public surface
# ---------------------------------------------------------------------------

EXPECTED_TOOL_NAMES = {
    "fact_get", "fact_add", "fact_profile", "fact_check", "fact_resolve",
    "fact_invalidate", "fact_relate", "fact_relationships", "fact_path",
    "fact_neighbors",
    "vault_read", "vault_write", "vault_overview", "vault_search",
    "vault_recall",
    "memory_read", "memory_add", "memory_replace", "memory_remove",
    "session_recall",
}


def test_list_tools_returns_expected_set():
    tools = asyncio.run(_aggregated_list_tools())
    names = {t.name for t in tools}
    missing = EXPECTED_TOOL_NAMES - names
    extra = names - EXPECTED_TOOL_NAMES
    assert not missing, f"missing tools: {missing}"
    assert not extra, f"unexpected tools: {extra}"
    assert len(tools) == len(EXPECTED_TOOL_NAMES), (
        f"duplicate tool registration? got {len(tools)} tools, "
        f"expected {len(EXPECTED_TOOL_NAMES)}"
    )


def test_list_tools_required_inputs_present():
    # Spot-check that critical tools declare their required params
    tools = asyncio.run(_aggregated_list_tools())
    by_name = {t.name: t for t in tools}

    assert by_name["fact_get"].inputSchema.get("required") == ["entity"]
    assert by_name["fact_add"].inputSchema.get("required") == ["entity", "category", "fact"]
    assert by_name["fact_relate"].inputSchema.get("required") == ["source", "target", "type"]
    assert by_name["vault_read"].inputSchema.get("required") == ["path"]
    assert by_name["vault_write"].inputSchema.get("required") == ["path", "content"]
    assert by_name["vault_search"].inputSchema.get("required") == ["query"]


# ---------------------------------------------------------------------------
# Module import smoke — regression guard for PR 1 extraction
# ---------------------------------------------------------------------------

def test_helpers_accessible_via_shared_module():
    # After PR 1 extraction these names live in agent_mcp._shared. The
    # cleanup PR (post-PR-4) deleted the agent_mcp.memory shim, so this
    # guard now pins the canonical home directly. If a future split
    # moves any of these out of `_shared`, update both the import alias
    # at the top of this file and this list together.
    required_names = [
        "_token_overlap", "_levenshtein", "_fuzzy_entity_match",
        "_parse_fact_frontmatter", "_write_fact_frontmatter",
        "_resolve_entity", "_find_entity_dir",
        "_load_aliases", "_save_aliases",
        "_get_entity_dirs_cached",
        "_ENTITY_STOPWORDS", "_SCORING_STOPWORDS", "_QUERY_STOPWORDS",
        "FACTS_ROOT", "ALIASES_PATH", "VAULT",
    ]
    for name in required_names:
        assert hasattr(memory, name), (
            f"agent_mcp._shared.{name} missing — split refactor lost a helper"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    # _token_overlap
    test_token_overlap_jaccard,
    test_token_overlap_empty,
    test_token_overlap_identical,
    test_token_overlap_case_insensitive,
    # _levenshtein
    test_levenshtein_classic,
    test_levenshtein_empty,
    test_levenshtein_identical,
    test_levenshtein_swap_order_symmetric,
    # _fuzzy_entity_match
    test_fuzzy_entity_match_exact_case_insensitive,
    test_fuzzy_entity_match_below_threshold,
    test_fuzzy_entity_match_length_ratio_block,
    test_fuzzy_entity_match_typo_too_strict,
    test_fuzzy_entity_match_separator_variation,
    test_fuzzy_entity_match_empty_candidates,
    # frontmatter
    test_parse_fact_frontmatter_no_frontmatter,
    test_parse_fact_frontmatter_unclosed,
    test_parse_fact_frontmatter_empty_block,
    test_parse_fact_frontmatter_round_trip,
    test_write_fact_frontmatter_starts_with_marker,
    # stopwords
    test_entity_stopwords_basic_membership,
    test_scoring_stopwords_superset_of_entity,
    test_query_stopwords_superset_of_entity,
    # paths
    test_path_constants_resolved_under_home,
    # tool surface
    test_list_tools_returns_expected_set,
    test_list_tools_required_inputs_present,
    # extraction smoke
    test_helpers_accessible_via_shared_module,
]


def main() -> int:
    passed = 0
    failed: list[tuple[str, str]] = []
    for t in _TESTS:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"  FAIL  {t.__name__}: {e!r}")

    print()
    print(f"{passed}/{len(_TESTS)} passed")
    if failed:
        print()
        print("Failures:")
        for name, err in failed:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
