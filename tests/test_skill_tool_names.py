"""Skills must not name tools that do not exist.

The 2026-09-04 tool-choice investigation traced Lloyd's habit of shelling out
to `curl` back to here. The skill literally named `websearch` — the one the
prefetcher injected on any web-shaped message — instructed `web_search` and
`web_fetch`. Neither has ever existed in Lloyd; the tools are `http_search`
and `http_fetch`. The call failed with an unknown-tool error, and the same
skills named Bash + curl as the recovery path.

That is a class of defect, not one typo: an auto-generated skill can mint a
plausible tool name at any time, and nothing checked. This test is the check.

Scope note: the vault carries older drift in the nightly-* skills
(`mem_get`, `mem_write`, `delegate_task`, `execute_code`). Those are recorded
in KNOWN_UNFIXED rather than silently allowed — they are real and should be
cleaned up, but they are outside the web-search-and-fetch fix and holding the suite red
on them would just get the test disabled. Anything NOT in that set fails
immediately, which is what stops a regression.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SKILLS_DIRS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]

# Tool names that were never real in Lloyd. Every one of these was found in an
# active skill on 2026-09-04.
PHANTOM_TOOLS = {
    "web_search", "web_fetch", "web_extract",
    "WebSearch", "WebFetch", "HTTPFetch",
    "mcp____http_search", "mcp____http_fetch",
    "mem_get", "mem_write", "mem_search",
    "delegate_task", "execute_code", "sessions_spawn", "skills_get", "skills_list",
    "write_file", "file_write", "file_edit", "file_read", "read_file",
    "vault_get", "run_bash", "search_files", "add_fact",
    "skill_view",
}

# `terminal` was the OpenClaw name for Bash. It is also an ordinary English
# word ("run it from the terminal"), so only tool-shaped usage counts:
# a backticked name, or call syntax.
_TERMINAL_AS_TOOL = re.compile(r"`terminal`|\bterminal\s*\(")

# The debt ledger is empty: every name above is now banned outright, and the
# 91 skills that carried one have been rewritten onto the real tool or
# archived. Keep it that way — a new entry here means a regression, not a
# grandfathering.
KNOWN_UNFIXED: set[str] = set()

# Skills allowed to write the phantom names down, because their job is to say
# these names are not real: the web-search-and-fetch skill tells the model so directly,
# and the two mining skills cite them as the worked example of why a generated
# skill must have its tool names validated before install.
ALLOWED_TO_MENTION = {
    "web-search-and-fetch",
    "nightly-skills-management",
    "trajectory-skill-mining",
}


def _active_skill_files() -> list[Path]:
    """Every SKILL.md the prompt actually advertises.

    Mirrors prompt_builder._load_skills_index: dot-prefixed directories are
    the archive and are excluded, as are quarantined skills.
    """
    from prompt_builder import _is_quarantined_skill

    out: list[Path] = []
    seen: set[str] = set()
    for root in SKILLS_DIRS:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in seen:
                continue
            skill_file = entry / "SKILL.md"
            if skill_file.exists() and not _is_quarantined_skill(skill_file):
                out.append(skill_file)
                seen.add(entry.name)
    return out


@pytest.fixture(scope="module")
def skill_files() -> list[Path]:
    files = _active_skill_files()
    if not files:
        pytest.skip("no skills directory on this machine")
    return files


def test_no_active_skill_names_a_phantom_tool(skill_files):
    """The regression that started it all: a skill telling the model to call
    a tool the aggregator has never advertised."""
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(PHANTOM_TOOLS))) + r")\b")
    offenders: dict[str, list[str]] = {}
    for path in skill_files:
        if path.parent.name in ALLOWED_TO_MENTION:
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        hits = sorted(set(pattern.findall(body)))
        if _TERMINAL_AS_TOOL.search(body):
            hits.append("terminal")
        if hits:
            offenders[path.parent.name] = hits
    assert offenders == {}, (
        "active skills name tools that do not exist: "
        f"{offenders}. The real tools are http_search / http_fetch / http_request."
    )


@pytest.mark.skipif(
    os.environ.get("LLOYD_SKIP_LIVE_MCP") == "1",
    reason="live aggregator discovery disabled",
)
def test_phantom_list_is_actually_phantom():
    """Guard the guard: if any name in PHANTOM_TOOLS ever becomes a real tool,
    this test must be updated rather than continuing to ban it."""
    import asyncio

    from agent_mcp import main as agent_main

    real = {t.name for t in asyncio.run(agent_main.list_tools())}
    wrongly_banned = PHANTOM_TOOLS & real
    assert wrongly_banned == set(), (
        f"these are real tools and must not be in PHANTOM_TOOLS: {wrongly_banned}"
    )


def test_web_lookup_skill_exists_and_names_the_real_tools():
    """The replacement for the archived `websearch` skill must be present and
    correct, or the phantom-name fix has no positive half."""
    candidates = [d / "web-search-and-fetch" / "SKILL.md" for d in SKILLS_DIRS]
    found = [p for p in candidates if p.exists()]
    if not found:
        pytest.skip("web-search-and-fetch skill not installed on this machine")
    body = found[0].read_text(encoding="utf-8", errors="replace")
    for tool in ("http_search", "http_fetch", "http_request"):
        assert tool in body, f"web-search-and-fetch must document {tool}"
    # The localhost carve-out has to stay: Bash + curl is correct there.
    assert "localhost" in body
