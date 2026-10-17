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

The last section checks the other kind of claim a docstring makes about the
system: a MEASUREMENT quoted as prose. A tool name that stopped existing fails
the call; a ledger cohort that stopped being true fails nothing, it just
misleads the next reader who trusts it and does not re-run the join. Both are
prose that has to be held against the code it describes, so both live here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app import prompt_builder as pb
from agent_mcp import code_graph as CG

ROOT = Path(pb.__file__).resolve().parents[1]
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


# ── the skill must never point a round at the SHARED live stash stack ─────────
#
# #1038. "Before starting" step 3 read: "The live tree must be clean. Uncommitted
# work is somebody else's in-flight change — stash it and say so, or wait." The
# first sentence has been false since 649193f ("the tree is shared — rebase and
# retest instead of refusing"): `round.py` records live dirt instead of refusing
# it, and gate/promote refuse only when a dirty path is one the round also
# changes. The second sentence was the hazard. The live checkout's stash stack is
# ONE global LIFO list shared by every author, and on 2026-09-11 a round
# implementing an unrelated item popped #573's recovered 136-line diff out of it,
# so the only copy survived as a hand-written re-stash. A second pop with nobody
# watching is a permanent loss — which is why the instruction, not just the
# wording, had to go. Vault e86ef0a7 reworded step 3; this is the pin that keeps
# it reworded.

STASH_INSTRUCTION = re.compile(
    r"stash it\b|commit or stash|stash the live edit|\bgit stash\b", re.I)


def _before_starting_item(n: int) -> str:
    """Item `n` of the skill's "Before starting" list, exactly as written."""
    body = VAULT_SKILL.read_text()
    start, end = body.find("## Before starting"), body.find("## Procedure")
    assert -1 not in (start, end), "the skill lost its 'Before starting' section"
    parts = re.split(r"(?m)^(\d+)\.\s", body[start:end])
    items = {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    assert n in items, f"'Before starting' has no step {n}: {sorted(items)}"
    return items[n]


@pytest.mark.live_vault
@pytest.mark.skipif(not VAULT_SKILL.exists(), reason="vault not present")
def test_the_skill_step3_never_tells_a_round_to_use_the_live_stash_stack():
    """Runs on every pytest invocation where the skill file exists on disk.

    `live_vault` is the file's existing convention for vault-reading tests and
    only keeps them off the automod gate's hard rung (`-m "not live_vault"`),
    where a prompt promotion landing an hour later would fail an unrelated round
    for the previous writer's edit. The skip that could hide this pin is the one
    that fires when `~/obsidian` is genuinely absent.
    """
    step3 = _before_starting_item(3)
    assert "stash" not in step3.lower(), (
        "step 3 tells a round to write the shared live tree's stash stack, which "
        f"one round has already eaten a recovered diff: {step3}")
    # Take the instruction out and the behaviour must stay, or the edit is
    # amnesia rather than a fix: disjoint dirt is tolerated, overlap is not.
    low = step3.lower()
    assert "tolerat" in low and "proceed" in low, (
        f"step 3 no longer says disjoint live dirt is tolerated: {step3}")
    assert "stop and report" in low, (
        f"step 3 no longer says an overlapping path means stop and report: {step3}")


@pytest.mark.live_vault
@pytest.mark.skipif(not VAULT_SKILL.exists(), reason="vault not present")
def test_the_skill_carries_no_stash_instruction_but_keeps_the_prohibition():
    body = VAULT_SKILL.read_text()
    hits = STASH_INSTRUCTION.findall(body)
    assert not hits, f"the skill instructs a stash write somewhere: {hits}"
    # The prohibition and its incident stay; only the imperative goes. Delete the
    # explanation and the next author re-derives the pop instead of avoiding it.
    assert "stash stack" in body.lower(), \
        "the skill lost the rule that the live stash stack is read-only to a round"


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


# ── prose that quotes a ledger cohort quotes the CURRENT measurement ──────────
#
# The same kind of failure in the other direction: a claim about code that has
# quietly stopped being true. The attribution rule for a `vault_land` row — a row
# vouches for the item IT names, and the harness-stamped session decides only a
# row that names none — is justified in prose by how many rows carry no
# `item_id`. That number moved twice in four weeks and the docstrings kept the
# old one each time, so the rule stood explained by a corpus nobody could go back
# and check: the cohort measured 2026-09-21 survived in four places and the one
# measured 2026-10-14 in another (the strings are the ban-list below, quoted
# nowhere else in the tree), and the arm each of them justified — an unconditional
# `or session_id` — was still closing items on OTHER items' landings when #2913
# narrowed it (`workers/sources/autocode.py:_vault_commits_since`, #2039 closed on
# #2027's vault sha).

#: The cohort as measured on 2026-10-17 over `promotions.jsonl` and its two
#: monthly archives: 870 `vault_land` rows, 681 of them carrying an `item_id`
#: across 344 distinct sessions, 189 carrying none, 133 of those a session.
ATTRIBUTION_COHORT_DATE = "2026-10-17"
RETIRED_ATTRIBUTION_FIGURES = ("56 of the 172", "86 of the 403", "of the 863",
                               "674 of those", "341 distinct sessions")
#: The rule's own surfaces: the two readers, the tool boundary that stamps the
#: session, and the two suites that pin both sides of the rule.
ATTRIBUTION_SOURCES = ("workers/sources/autocode.py", "scripts/automod/backlog.py",
                       "agent_mcp/automod.py")
ATTRIBUTION_TESTS = ("tests/test_automod_review.py", "tests/test_vault_surface_churn.py")

_QUOTED_COHORT = re.compile(r"(?:\d+ )?of the [\d,]{3,}\b")


def _prose_blocks(rel: str) -> list[str]:
    """Every docstring and every run of whole-line comments in one source file.

    Docstrings only would miss `agent_mcp/automod.py`, where the attribution
    comment that rotted is a comment run inside the tool handler."""
    import ast
    src = (ROOT / rel).read_text(encoding="utf-8")
    blocks = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                blocks.append(doc)
    run: list[str] = []
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            run.append(line)
        elif run:
            blocks.append("\n".join(run))
            run = []
    if run:
        blocks.append("\n".join(run))
    return blocks


def test_no_attribution_prose_cites_a_retired_vault_land_cohort():
    """The retired cohorts stay retired. Each string in the ban-list is one that
    sat in the tree and read as a live measurement, because the prose quoting it
    carried no date a later reader could contradict.

    This file is the one place allowed to name them — and it names each exactly
    once, in the constant, which is what keeps the list checkable rather than
    padded and stops the ban-list itself becoming a citation nobody re-runs."""
    here = Path(__file__).resolve()
    for path in [*(ROOT / rel for rel in ATTRIBUTION_SOURCES),
                 *(ROOT / rel for rel in ATTRIBUTION_TESTS),
                 *sorted((ROOT / "tests").glob("test_*.py"))]:
        text = path.read_text(encoding="utf-8")
        for retired in RETIRED_ATTRIBUTION_FIGURES:
            if path.resolve() == here:
                assert text.count(retired) == 1, (
                    f"the ban-list names the retired cohort {retired!r} "
                    f"{text.count(retired)} times; it belongs in the constant only")
                continue
            assert retired not in text, (
                f"{path.relative_to(ROOT)} still cites the retired vault_land "
                f"cohort {retired!r}; re-measure it against the promotions ledger")


def test_vault_land_attribution_prose_cites_the_measured_cohort_with_its_date():
    """Every block that quotes a fraction of the `vault_land` rows quotes the
    measurement WITH the date it was taken, so the next reader re-runs the join
    instead of trusting a number that has since moved."""
    checked = 0
    for rel in (*ATTRIBUTION_SOURCES, *ATTRIBUTION_TESTS):
        for block in _prose_blocks(rel):
            if "`vault_land`" not in block or not _QUOTED_COHORT.search(block):
                continue
            checked += 1
            assert ATTRIBUTION_COHORT_DATE in block, (
                f"{rel}: a `vault_land` cohort quoted without {ATTRIBUTION_COHORT_DATE} "
                f"is an undated measurement — {block[:120]!r}")
    assert checked >= 6, (
        f"only {checked} attribution blocks were checked: the scan matched almost "
        "nothing, which is how a prose check passes by testing nothing")
