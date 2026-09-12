"""Every `graph_*` named in a prompt, skill or doc is a tool that exists.

A tool the model is told to call and cannot is worse than one it was never
told about: it spends a turn on the call, gets "Unknown tool", and has no
way to tell a typo from a disabled module. These claims live in five
different places — the system prompt, the implementer's worker prompt, a
vault skill, CLAUDE.md and architecture/tools.md — and only one of them is
in the same file as the tool list.

The ordering and placement assertions exist for the same reason the
paragraph does: on 2026-09-04 the web-lookup investigation found that the
model shells out to curl because Bash has an affordance paragraph and
http_* did not. Naming a tool is not enough; *where* it is named is part of
whether it gets used.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import prompt_builder as pb
from agent_mcp import code_graph as CG

ROOT = Path(pb.__file__).resolve().parent
VAULT_SKILL = Path.home() / "obsidian" / "skills" / "automod-change-own-code" / "SKILL.md"

GRAPH_TOOL_RE = re.compile(r"\bgraph_[a-z_]+")


async def _advertised() -> set[str]:
    return {t.name for t in await CG.list_tools()}


def _system_prompt() -> str:
    return pb.build_system_prompt(include_skills_index=False)


def _implement_prompt() -> str:
    from workers.sources import autocode
    return autocode.PROMPT


async def test_every_graph_name_in_the_system_prompt_exists():
    names = _extract(_system_prompt())
    assert names, "the system prompt names no graph tool at all"
    assert names <= await _advertised(), names - await _advertised()


async def test_every_graph_name_in_the_implement_prompt_exists():
    names = _extract(_implement_prompt())
    assert names <= await _advertised(), names - await _advertised()


async def test_every_graph_name_in_claude_md_and_tools_md_exists():
    advertised = await _advertised()
    for rel in ("CLAUDE.md", "architecture/tools.md"):
        names = _extract((ROOT / rel).read_text())
        assert names <= advertised, f"{rel}: {names - advertised}"


@pytest.mark.live_vault
@pytest.mark.skipif(not VAULT_SKILL.exists(), reason="vault not present")
async def test_every_graph_name_in_the_automod_skill_exists():
    names = _extract(VAULT_SKILL.read_text())
    assert names <= await _advertised(), names - await _advertised()


def _extract(text: str) -> set[str]:
    return set(GRAPH_TOOL_RE.findall(text))


# ── placement, not just presence ────────────────────────────────────────────

def test_system_prompt_puts_code_navigation_before_turn_discipline():
    s = _system_prompt()
    nav, turn = s.find("Code navigation:"), s.find("Turn discipline:")
    assert nav != -1, "no code-navigation affordance paragraph"
    assert turn != -1
    assert nav < turn, "code navigation must come before turn discipline"


def test_system_prompt_keeps_grep_correct_for_what_the_graph_cannot_see():
    """Overshoot is the risk of naming a new tool; say what it is blind to."""
    s = _system_prompt()
    para = s[s.find("Code navigation:"):s.find("Turn discipline:")]
    assert "Grep" in para
    assert "seam" in para or "HTTP" in para


def test_system_prompt_paragraph_carries_no_per_turn_value():
    """A per-turn value in the system prompt re-prefills it every turn."""
    s = _system_prompt()
    para = s[s.find("Code navigation:"):s.find("Turn discipline:")]
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", para)
    assert "SM_20" not in para


def test_implement_prompt_sends_the_round_to_the_skill_that_maps_the_radius():
    """Since cut 4 of senses-not-supervision the prompt is the contract and
    the procedure — the blast-radius step included — lives in the vault skill
    it names. The two tests below pin the step there."""
    p = _implement_prompt()
    assert "automod-change-own-code" in p
    assert "graph_affected" not in p, "procedure crept back into the prompt"


@pytest.mark.live_vault
@pytest.mark.skipif(not VAULT_SKILL.exists(), reason="vault not present")
def test_the_skill_asks_for_the_blast_radius_with_a_worktree_root():
    s = VAULT_SKILL.read_text()
    assert "graph_affected" in s
    assert "root=<worktree>" in s, \
        "without root= the implementer maps the live checkout, not its own"


@pytest.mark.live_vault
@pytest.mark.skipif(not VAULT_SKILL.exists(), reason="vault not present")
def test_the_skill_procedure_stays_numbered_in_order():
    """The blast-radius step was inserted as step 2; the ones after it had to
    move, and a later edit must not leave a gap."""
    s = VAULT_SKILL.read_text()
    start = s.find("## Procedure")
    end = s.find("## The round contract")
    assert -1 not in (start, end)
    nums = [int(m) for m in re.findall(r"^(\d+)\. ", s[start:end], re.M)]
    assert nums == list(range(1, len(nums) + 1)) and len(nums) >= 6, nums
    assert "graph_affected" in s[start:end]


PATCH = "scripts/maintenance/vault-automod-skill-blast-radius.patch"
APPLY = f"git -C ~/obsidian apply {PATCH}"


@pytest.mark.live_vault
@pytest.mark.skipif(not VAULT_SKILL.exists(), reason="vault not present")
def test_automod_skill_maps_the_radius_between_opening_and_working():
    """The vault half of this change, and the reason it is a separate step.

    The vault is a live shared tree with no PR path, so editing it lands
    immediately — while the `graph_*` tools it names only exist after this
    branch merges and lloyd-mcp restarts. An edited skill therefore tells
    every automod round in the gap to call a tool that returns "Unknown
    tool", and makes quality gate 4 unsatisfiable. So the edit is held as a
    patch and applied at merge time, and THIS test is the reminder.

    `live_vault` keeps it off the automod gate's hard rung (`-m "not
    live_vault"`), where a vault someone else rewrote would fail an
    unrelated round.
    """
    body = VAULT_SKILL.read_text()
    proc = body[body.find("## Procedure"):body.find("## What the gate rejects")]
    open_at = proc.find("**Open a round.**")
    map_at = proc.find("**Map the blast radius.**")
    work_at = proc.find("**Do the work normally.**")
    assert -1 not in (open_at, map_at, work_at), (
        f"the vault skill has no blast-radius step yet — apply it with:\n"
        f"    {APPLY}\n"
        f"Do this AFTER lloyd-mcp restarts with code_graph, not before: the "
        f"step names tools that do not exist until then."
    )
    assert open_at < map_at < work_at


@pytest.mark.live_vault
@pytest.mark.skipif(not VAULT_SKILL.exists(), reason="vault not present")
def test_automod_skill_procedure_stays_numbered_in_order():
    """Holds before and after the patch — the insertion renumbers 6-8 to 7-9."""
    body = VAULT_SKILL.read_text()
    proc = body[body.find("## Procedure"):body.find("## What the gate rejects")]
    nums = [int(m) for m in re.findall(r"^(\d+)\. \*\*", proc, re.M)]
    assert nums == list(range(1, len(nums) + 1)), nums


# ── the tools the model is handed by default ────────────────────────────────

def test_baseline_tools_carry_the_two_navigation_tools():
    """With ToolSearch on, a tool outside the baseline must be searched for
    before it can be called — which is one iteration the model will not
    spend on a habit it does not have yet."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    baseline = cfg["harness"]["tool_search"]["baseline_tools"]
    assert "graph_explain" in baseline
    assert "graph_affected" in baseline


def test_code_graph_config_block_exists_and_is_complete():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    block = cfg["code_graph"]
    for key in ("graphify_bin", "auto_refresh", "refresh_timeout_s",
                "min_refresh_interval_s", "max_cached_roots", "max_lines"):
        assert key in block, key
    assert "enabled" not in block, \
        "an enabled flag that emptied list_tools() breaks annotation staleness"


def test_docs_describe_the_module():
    assert "## Code graph" in (ROOT / "CLAUDE.md").read_text()
    tools_md = (ROOT / "architecture" / "tools.md").read_text()
    assert "| `code_graph` |" in tools_md
    assert "### Code graph" in tools_md


def test_the_eval_set_exists_and_has_controls():
    """A graph-shaped metric with no control rows scores overshoot as success."""
    import yaml
    spec = yaml.safe_load((ROOT / "eval" / "code_nav_queries.yaml").read_text())
    qs = spec["queries"]
    nav = [q for q in qs if q["category"] == "code-nav"]
    controls = [q for q in qs if q["category"].startswith("control")]
    assert len(nav) >= 10 and len(controls) >= 5
    for q in nav:
        assert all(t.startswith("graph_") for t in q["expect_tools"]), q["id"]
    for q in controls:
        assert not any(t.startswith("graph_") for t in q["expect_tools"]), q["id"]
