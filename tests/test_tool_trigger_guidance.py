"""Tool descriptions must carry trigger conditions that survive the catalog.

Backlog #361. An AST audit of every `Tool(name=…, description=…)` in
`agent_mcp/*.py` found 26 of 98 descriptions with any conditional or
cross-tool phrasing — the triage recorded the same shape as 14 of 92 before
main grew more tools — and the item's own priority list was the worst case:
12 of the 20 tools named here opened with purely mechanical "what it does"
text. Tool choice is where a description earns its tokens, so a description
that only names the mechanism leaves the model to guess the boundary.

The naive fix is useless on its own. Deferred tools reach the model as a
one-line gist: `app/harness/tool_search.py:_gist()` takes the first sentence
and hard-caps it at `CATALOG_GIST_CHARS`, and the full description only
arrives after a ToolSearch round-trip — after the choice was already made.
#418 tracks that mechanism, and `harness.tool_search.enabled` is currently
false (#456), so today every schema is handed over in full and the gist is
latent. The convention below is what makes the rewrite hold either way:
**the trigger clause is the first sentence, and it fits inside the cap.**
Written any other way the guidance sits below the fold of the summary line —
which is how `mc_navigate` (1707 chars, the strongest trigger rule in the
codebase) shipped for months as "Switch the user's Mission Control tab …".

Three invariants, checked against the real `_gist` and the real token
estimator rather than copies of them:

1. Every priority tool's description states a condition AND names a real
   alternative tool — a "when NOT to use" with no safer option disambiguates
   nothing.
2. `_gist(description)` still contains both. This is the part that makes the
   rewrite worth doing: guidance the model only sees after loading the schema
   was not available when it picked the tool.
3. The catalog reminder the rewrite costs stays under the ~5k tokens
   `tool_search.py:209-212` records as the reason truncation exists.

The coverage floor is a regression guard, not an aspiration. Before this
change invariant 1 failed outright on 12 of these 20 tools and the
whole-catalog count was 26.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENT_MCP = ROOT / "agent_mcp"


def _lit(node) -> str:
    """Literal string of a description node, f-strings folded flat.

    Nine descriptions are f-strings interpolating a constant (a default
    timeout, a path). Dropping the interpolated value still leaves the prose,
    which is what the audit is about; refusing to read them would hide nine
    tools from the coverage count.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return " ".join(
            _lit(v) for v in node.values if not isinstance(v, ast.FormattedValue)
        )
    return ""


def _tool_descriptions() -> dict[str, str]:
    """{name: description} for every literal ``Tool(name=…, description=…)``."""
    found: dict[str, str] = {}
    for path in sorted(AGENT_MCP.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # `Tool(...)` and `types.Tool(...)` are the same call.
            if (getattr(node.func, "id", None)
                    or getattr(node.func, "attr", None)) != "Tool":
                continue
            kw = {k.arg: k.value for k in node.keywords}
            if "name" not in kw or "description" not in kw:
                continue
            name, desc = _lit(kw["name"]), _lit(kw["description"])
            if name and desc:
                found.setdefault(name, desc)
    return found


ALL_TOOL_NAMES = set(_tool_descriptions())

# The set #361 itself prioritised (Bash, the http trio, the vault tools, the
# file tools, Task), extended with the pairs those boundaries actually run
# against: Read is only decidable next to Glob/Grep/Write/Edit, vault_search
# next to vault_read/vault_recall, mc_navigate next to mc_get_state/
# mc_close_modal, and the IDE trio is the other half of every "show me X"
# request the mc pair answers.
PRIORITY_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "Task",
    "vault_read",
    "vault_write",
    "vault_search",
    "vault_recall",
    "http_search",
    "http_fetch",
    "http_request",
    "mc_navigate",
    "mc_close_modal",
    "mc_get_state",
    "ide_open_file",
    "ide_open_folder",
    "ide_close_tab",
]

# Conditional phrasing: the description tells the model WHEN, not just WHAT.
CONDITIONAL = re.compile(
    r"\b(use (?:this |it )?(?:when|whenever|only|to|for|first)"
    r"|when to use|before (?:calling|using|you)|instead of|rather than"
    r"|do not use|don't use|never use|not for|prefer|reach for|unless"
    r"|use [a-z_]+ (?:first|instead)|then [a-z_]+ )",
    re.I,
)


def _named_tools(text: str, *, exclude: str = "") -> list[str]:
    """Real tool names appearing in `text` — not prose that sounds like one."""
    return sorted(
        other for other in ALL_TOOL_NAMES - {exclude}
        if re.search(rf"\b{re.escape(other)}\b", text)
    )


@pytest.fixture(scope="module")
def descriptions() -> dict[str, str]:
    return _tool_descriptions()


def test_audit_sees_the_tools_it_claims(descriptions):
    """If the audit can't find them, every assertion below is vacuous."""
    missing = [n for n in PRIORITY_TOOLS if n not in descriptions]
    assert missing == [], f"priority tools not found as literal Tool(): {missing}"


@pytest.mark.parametrize("name", PRIORITY_TOOLS)
def test_priority_tool_states_a_trigger_condition(descriptions, name):
    desc = descriptions[name]
    assert CONDITIONAL.search(desc), (
        f"{name}: description is purely functional — no 'when to use' "
        f"condition. Opens: {desc[:120]!r}"
    )


@pytest.mark.parametrize("name", PRIORITY_TOOLS)
def test_priority_tool_names_a_competing_tool(descriptions, name):
    desc = descriptions[name]
    assert _named_tools(desc, exclude=name), (
        f"{name}: no cross-tool pointer — the description never names the "
        f"tool to reach for instead, so it cannot settle a tool choice."
    )


@pytest.mark.parametrize("name", PRIORITY_TOOLS)
def test_trigger_survives_the_catalog_gist(descriptions, name):
    """The whole point of the convention.

    A trigger clause past the first sentence, or longer than the cap, is
    text the model reads only after it already chose the tool.
    """
    from app.harness.tool_search import _gist

    gist = _gist(descriptions[name])
    assert not gist.endswith("…"), (
        f"{name}: the catalog gist is hard-capped mid-clause — {gist!r}. "
        f"Shorten the leading trigger sentence."
    )
    assert CONDITIONAL.search(gist), (
        f"{name}: gist lost the trigger condition — {gist!r}"
    )
    assert _named_tools(gist, exclude=name), (
        f"{name}: gist lost the alternative tool — {gist!r}"
    )


def test_trigger_coverage_rose_materially(descriptions):
    """"Materially from the 26 measured before this change" — the number
    #361's acceptance names as a rise, not a spot-check.

    The rewrite has to spread or the priority set is window dressing over a
    catalog that still says nothing.
    """
    guided = [n for n, d in descriptions.items() if CONDITIONAL.search(d)]
    assert len(guided) >= 32, (
        f"only {len(guided)} of {len(descriptions)} descriptions carry a "
        f"trigger condition (baseline 26)"
    )


def test_catalog_reminder_stays_under_its_token_ceiling(descriptions):
    """The ~5k note at tool_search.py:209-212 is why `_gist` exists.

    Trigger-first descriptions are paid on every request while progressive
    disclosure is on, so the size is measured rather than assumed. Ceiling:
    the ~5k tokens the comment records as unacceptable for whole descriptions.
    """
    from app.compaction import estimate_tokens
    from app.harness.tool_search import format_catalog_reminder

    catalog = [
        {"function": {"name": n, "description": d}}
        for n, d in sorted(descriptions.items())
    ]
    tokens = estimate_tokens(format_catalog_reminder(catalog))
    assert tokens < 5000, (
        f"catalog reminder now costs {tokens} tokens — at or past the ~5k "
        f"ceiling that motivated truncation"
    )
