"""Skills must not name tools that do not exist.

The 2026-09-04 tool-choice investigation traced Lloyd's habit of shelling out
to `curl` back to here. The skill literally named `websearch` — the one the
prefetcher injected on any web-shaped message — instructed `web_search` and
`web_fetch`. Neither has ever existed in Lloyd; the tools are `http_search`
and `http_fetch`. The call failed with an unknown-tool error, and the same
skills named Bash + curl as the recovery path.

That is a class of defect, not one typo: an auto-generated skill can mint a
plausible tool name at any time, and nothing checked. This test is the check.

Two classes sat outside the denylist and stayed invisible (item #410), and a
denylist is structurally unable to see either:

  * a phantom named in **prose or call syntax** rather than as a bare token.
    `Agent` is the subagent tool of other harnesses; Lloyd's is `Task`. The
    registry is the authority and `test_phantom_list_is_actually_phantom` asks
    it on every run, so nothing here has to assert how many tools are served or
    that `Agent` was never one of them. Five active skills use it as a tool
    anyway — `deep-research`, `research-agent`, `strict-task-mapping` and
    `subagent-orchestrate` instruct the call, and `discord-social` forbids it,
    which is correct prose that any matcher has to survive. Because `Agent` is
    also ordinary English, and the literal `author: Hermes Agent (adapted from
    obra/superpowers)` front matter of every Hermes-authored skill, it goes in
    `_AGENT_AS_TOOL` with a tool-shaped matcher rather than into
    `PHANTOM_TOOLS` — the rule `_TERMINAL_AS_TOOL` already applies to
    `terminal`.
  * a **retired** name, which is neither real nor in the denylist. The
    `selfmod_*` family became `automod_*`; a skill written against the old
    names passes a set-membership test forever, which is why #419's own seed set
    prescribed calls that no longer exist. `_RETIRED_PREFIX` matches the family
    by prefix instead, so closing it is not an enumeration problem. The corpus
    holds zero `selfmod_` occurrences today — this matcher exists for the next
    mined skill, not for a live breakage.

Scope note, re-measured for #410: the older drift this file was written around
— `mem_get`, `mem_write`, `delegate_task`, `execute_code` in the nightly-*
skills — is gone from the corpus. `mem_get` survives only in
`nightly-skill-consolidation`, which is in `ALLOWED_TO_MENTION` because it cites
the name as the worked example of why a generated skill must validate its tool
names; the other three are named by no active skill at all. `KNOWN_UNFIXED` is
therefore empty and, unlike the path ledger below, nothing in this file reads
it: a phantom name is banned outright and a new one fails immediately. Treat an
entry added here as a regression, not a grandfathering — and if you are looking
for the ledgers that actually exempt something, they are `PATH_KNOWN_UNFIXED`
and `AGENT_MENTION_EXEMPT`, both of which the tests below do consult.
"""

from __future__ import annotations

import os
import re
import subprocess
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
    "file_glob", "file_grep", "tag_search", "tag_explore",
    "pipeline_dispatch", "chat_send", "sessions_send",
}

# `terminal` was the OpenClaw name for Bash. It is also an ordinary English
# word ("run it from the terminal"), so only tool-shaped usage counts:
# a backticked name, or call syntax.
_TERMINAL_AS_TOOL = re.compile(r"`terminal`|\bterminal\s*\(")

# `Agent` is the subagent-dispatch tool of other harnesses; Lloyd's is `Task`
# (`agent_mcp.main.list_tools()` serves `Task`, never `Agent`). It is also an
# ordinary English word, the `author: Hermes Agent (adapted from
# obra/superpowers)` front matter of every Hermes-authored skill, and the
# literal `new https.Agent({ rejectUnauthorized: false })` of the TLS skill —
# so only tool-shaped usage counts, the same rule `_TERMINAL_AS_TOOL` applies
# to `terminal`: bolded prose, a backticked name, or a call. The lookbehind is
# what keeps `https.Agent(` out: a preceding word character or dot is an
# attribute, not a tool.
_AGENT_AS_TOOL = re.compile(r"\*\*Agent tool\*\*|`Agent`|(?<![\w.])Agent\(")

# A retired tool family: the name is neither served nor in the denylist, so set
# membership cannot see it however many retired names get written down.
# `selfmod_*` became `automod_*` after #419's seed set was authored to the old
# spelling; a skill written to that spelling today must not install green.
_RETIRED_PREFIX = re.compile(r"\bselfmod_[a-z0-9_]+")

# Skills allowed to carry an `Agent`-tool mention, each because an item already
# owns its removal or because the sentence is a prohibition. This is the only
# way to land a matcher for a class the corpus is already covered by (#410
# clause 5); every entry names its owner, and none may be added to for a skill
# that has no open item saying what to do with it.
AGENT_MENTION_EXEMPT: dict[str, str] = {
    # Archives this whole skill, so the mention leaves with the file.
    "subagent-orchestrate": "#409 clause 1",
    # Must name `Task` (or route to pipeline-dispatch) instead; the matcher's
    # offender list is what makes that visible instead of merely asserted.
    "deep-research": "#409 clause 3",
    "research-agent": "#409 clause 3",
    # Correct prose about a tool that genuinely does not exist: it *forbids*
    # the Agent tool. #409 clause 3 keeps this sentence intact, so it stays
    # exempt permanently, not until someone fixes it.
    "discord-social": "#409 clause 3 (prohibition, keep)",
    # `Agent`-tool dispatch in four places; named by no other item's clauses.
    "strict-task-mapping": "#410",
}

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
    "nightly-skill-consolidation",
    # Defines a plugin hook literally named chat_send in Python.
    "create-hermes-plugin",
    # Cites the names as the worked example of wrong-tool-name diagnosis.
    "autonomy-task-diagnosis",
    # Says in its own text that pipeline_dispatch is not a tool.
    "pipeline-dispatch",
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


_PHANTOM_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, sorted(PHANTOM_TOOLS))) + r")\b")


def _phantom_hits(body: str, *, skip_agent: bool = False) -> list[str]:
    """Every tool name in `body` that no session could ever have called.

    Three shapes, one per class of defect. A bare token from the denylist;
    `terminal` or `Agent` used as a tool (prose, backticks or call syntax);
    and any `selfmod_`-prefixed name, retired when that family became
    `automod_*`. This is the whole matcher — the corpus test and the fixture
    probes below both run it, so a probe can never drift from what production
    checks.

    `skip_agent` is how a skill gets excused for naming the Agent tool and for
    nothing else. An exemption has to be as narrow as the defect it excuses, or
    a skill in the debt ledger is free to mint a brand-new phantom name.
    """
    hits = sorted(set(_PHANTOM_PATTERN.findall(body)))
    if _TERMINAL_AS_TOOL.search(body):
        hits.append("terminal")
    if _AGENT_AS_TOOL.search(body) and not skip_agent:
        hits.append("Agent")
    retired = sorted(set(_RETIRED_PREFIX.findall(body)))
    if retired:
        hits.append(",".join(retired))
    return hits


def _phantom_offenders(files: list[Path], *,
                       agent_exempt: frozenset[str] = frozenset()
                       ) -> dict[str, list[str]]:
    """skill-directory-name → phantom names, minus the named exemption sets."""
    offenders: dict[str, list[str]] = {}
    for path in files:
        skill = path.parent.name
        if skill in ALLOWED_TO_MENTION:
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        hits = _phantom_hits(body, skip_agent=skill in agent_exempt)
        if hits:
            offenders[skill] = hits
    return offenders


def test_no_active_skill_names_a_phantom_tool(skill_files):
    """The regression that started it all: a skill telling the model to call
    a tool the aggregator has never advertised."""
    offenders = _phantom_offenders(skill_files,
                                   agent_exempt=frozenset(AGENT_MENTION_EXEMPT))
    assert offenders == {}, (
        "active skills name tools that do not exist: "
        f"{offenders}. The real tools are http_search / http_fetch / http_request, "
        "and the subagent tool is Task."
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
    # The two matchers added for #410 ban things the denylist does not name, so
    # the registry is the only independent check that they are not banning a
    # live tool: `Agent` by exact name (Lloyd's subagent tool is `Task`), and
    # the retired family by prefix, since the whole point of `_RETIRED_PREFIX`
    # is that no individual retired name is written down anywhere.
    assert "Agent" not in real, (
        "`Agent` has become a served tool; the `_AGENT_AS_TOOL` matcher and the "
        "AGENT_MENTION_EXEMPT debt must be retired with it"
    )
    retired_served = {t for t in real if _RETIRED_PREFIX.fullmatch(t)}
    assert retired_served == set(), (
        f"`selfmod_` is served again: {retired_served} — update _RETIRED_PREFIX"
    )


@pytest.mark.parametrize("line", [
    "Use the **Agent tool** to spawn a subagent.",
    "Send `Agent` calls in parallel for independent work.",
    "Agent({\n  prompt: \"do it\",\n  subagent_type: \"general-purpose\",\n})",
])
def test_an_agent_tool_mention_fails_the_guard(tmp_path, line):
    """A phantom named in prose, backticks or call syntax, not a bare token.

    The denylist could never see this, which is why five active skills still
    tell a session to call a tool that has never been served. Injecting the
    line into an ordinary active SKILL.md is exactly what this pins: the guard
    must report it. Remove the line and the same helper goes quiet — that
    half is the next test.
    """
    skill = tmp_path / "some-active-skill" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text("---\nstatus: active\n---\n# SKILL: some-active-skill\n\n",
                     encoding="utf-8")
    before = _phantom_offenders([skill])
    assert before == {}, "fixture must start clean or the probe proves nothing"
    skill.write_text(skill.read_text(encoding="utf-8") + line + "\n",
                     encoding="utf-8")
    offenders = _phantom_offenders([skill])
    assert offenders == {"some-active-skill": ["Agent"]}, (
        f"`{line[:40]}` is tool-shaped usage of a tool that does not exist and "
        f"the guard stayed silent: {offenders}"
    )


def test_the_agent_matcher_survives_the_words_it_is_not(tmp_path):
    """The matcher is tool-shaped, not word-shaped (#410 clause 2).

    `Agent` is ordinary English, the author line of every Hermes-authored
    skill, and an HTTP connector's class name. A matcher that fired on these
    would be deleted within a week, which is the failure mode the clause
    exists to prevent — so it is pinned on the two literal strings from the
    live corpus, not on invented lookalikes.
    """
    skill = tmp_path / "https-agent-client" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text(
        "---\nstatus: active\n---\n"
        "author: Hermes Agent (adapted from obra/superpowers)\n\n"
        "```js\nconst agent = new https.Agent({ rejectUnauthorized: false });\n"
        "```\n\n"
        "Run the agent loop, then read the agent's report.\n",
        encoding="utf-8")
    assert _phantom_offenders([skill]) == {}, (
        "the Hermes author front matter and `https.Agent(` must not read as "
        "dispatching a phantom tool"
    )


def test_a_retired_selfmod_tool_name_fails_the_guard(tmp_path):
    """A retired name is neither real nor in the denylist, so a set-membership
    test cannot flag it however many such names get added (#410 clause 3).

    The `selfmod_*` → `automod_*` rename means a skill written against the old
    spelling installs and runs green today. Matching the family by prefix is
    what stops that being an enumeration problem.
    """
    skill = tmp_path / "legacy-automod-habits" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text(
        "---\nstatus: active\n---\n"
        "Call `selfmod_gate` now, then `selfmod_land` once it reports clean.\n",
        encoding="utf-8")
    offenders = _phantom_offenders([skill])
    assert offenders == {
        "legacy-automod-habits": ["selfmod_gate,selfmod_land"]
    }, f"a retired tool name passed green: {offenders}"


def test_the_live_corpus_defects_are_all_the_named_ones(skill_files):
    """The exemption set is a debt ledger, so it needs an upper bound.

    The other half of #410 clause 5. Green under an exemption proves nothing on
    its own — the ledger could be bigger than the corpus, or shielding a skill
    that never had the defect, or shielding one class while the skill offends
    in another. So: run the scan with AGENT_MENTION_EXEMPT switched off (the
    pre-existing `ALLOWED_TO_MENTION` skills sit outside this ledger — they are
    exempt from every class and predate #410), and the Agent-class offenders
    must be *exactly* the keys of AGENT_MENTION_EXEMPT
    (each of which names the item that owns its removal), and no skill may carry
    a `selfmod_` name at all, since nothing is exempt from that matcher. A skill
    that minted a denylist phantom while sitting in the Agent ledger fails the
    main test instead, because these exemptions excuse one class and nothing
    else.
    """
    raw = _phantom_offenders(skill_files)
    agent_offenders = {n for n, hits in raw.items() if "Agent" in hits}
    retired_offenders = {n for n, hits in raw.items()
                         if any(h.startswith("selfmod_") for h in hits)}
    assert agent_offenders == set(AGENT_MENTION_EXEMPT), (
        "the Agent-tool-naming corpus and AGENT_MENTION_EXEMPT have diverged: an "
        "entry was added for a clean skill, or a skill was fixed and its "
        f"exemption left behind. offenders={sorted(agent_offenders)} "
        f"exempt=sorted({sorted(AGENT_MENTION_EXEMPT)})"
    )
    assert retired_offenders == set(), (
        f"active skills name retired selfmod_* tools: {sorted(retired_offenders)}"
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


# ─────────────────────────────────────────────────────────────────────────
# The same defect one level up: a skill naming a PATH that is not there.
# ─────────────────────────────────────────────────────────────────────────

VAULT = Path.home() / "obsidian"

#: Checked-out source a doc may name. Anything else under the repo root is
#: runtime state (`_pipeline/`, `sessions/`, `*.db`, logs), which a run creates
#: and a checkout does not have — naming it is not drift.
CHECKOUT_PREFIXES = ("agent-services/", "agent_mcp/", "app/", "architecture/",
                     "eval/", "tests/", "web/")

#: Skill-local subdirectories, which resolve inside the skill's own folder as
#: well as at the repo root (`powerpoint/scripts/office/soffice.py`).
SKILL_LOCAL_PREFIXES = ("scripts/", "references/", "assets/")

RUNTIME_FIRST = ("_pipeline/", "autonomy-runs/", "sessions/", "data/", "logs/",
                 ".venvs/", ".local/")
#: Host material a checkout never carries. Certificates and keys join the list
#: because `agent-services/cert/` is gitignored by design — a private key in git
#: is the thing the ignore exists to prevent — so a doc naming ca.crt is naming
#: a file on the machine, not drift. Before this, `system-health-check` naming
#: the CA it signs client certs with was reported as an unresolved reference in
#: every worktree, which is a red node at base rather than a fixable claim.
RUNTIME_SUFFIXES = (".db", ".sqlite", ".jsonl", ".log", ".csv", ".lock", ".pid",
                    ".crt", ".key", ".pem", ".srl")

_LLOYD_PATH = re.compile(r"(?<![\w/.~-])(?:~|/home/[A-Za-z0-9_.-]+)/lloyd/([A-Za-z0-9_.\-/*]+)")
_OBSIDIAN_PATH = re.compile(r"(?<![\w/.~-])(?:~|/home/[A-Za-z0-9_.-]+)/obsidian/([A-Za-z0-9_.\-/*]+)")
_FENCE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.S)
BACKTICKED = re.compile(r"`([^`\n]+)`")
# A .py path inside a fenced command block: prose is conservative, but a
# command block that names a script means "run this script".
_PY_IN_COMMAND = re.compile(r"(?:^|[\s/])((?:eval|app|tests|scripts)/[A-Za-z0-9_./-]+\.py)")

#: Every unresolved reference that existed on 2026-09-18, when this check was
#: written, across 230 skill/task files and 381 references. Same contract as
#: `KNOWN_UNFIXED` above: real drift, each one worth fixing, held out of the
#: assertion so an unrelated round is not blocked by prose it did not touch. A
#: NEW entry here is a regression; `autonomy/85-*` and `secondary-routing-eval`
#: are deliberately NOT in it — they are what item #1240 fixed, and
#: `test_the_secondary_routing_nightly_docs_are_clean` keeps them out.
PATH_KNOWN_UNFIXED: set[str] = {
    "skills/ai-engineer-monitor/SKILL.md::vault:autonomy/75-ai-engineer-youtube-monitor.md",
    "skills/browser-session-extract/SKILL.md::vault:skills/browser-session-extract/browser_session_extract.py",
    "skills/deep-research/SKILL.md::repo:scripts/vault/okf_taxonomy.CANONICAL_TYPES",
    "skills/documentation-digester/SKILL.md::repo:agent-services/llm/llama.cpp",
    "skills/documentation-digester/SKILL.md::repo:agent-services/llm/llama.cpp/build/bin/llama-server",
    "skills/entity-resolution-sweep/SKILL.md::repo:agent_mcp/memory.py",
    "skills/file-path-resolution/SKILL.md::repo:inner-voice/system_prompt.md",
    "skills/file-path-resolution/SKILL.md::repo:lloyd/inner-voice/system-prompt.md",
    "skills/file-path-resolution/SKILL.md::vault:lloyd/inner-voice/system-prompt.md",
    "skills/file-processor/SKILL.md::vault:skills/file-processor/file_processor.py",
    "skills/file-read-resilience/SKILL.md::vault:agents/idler/gateway.py",
    "skills/historical-knowledge-refresh/SKILL.md::repo:scripts/memory/extract-session-log.py",
    "skills/iv-plan-review/SKILL.md::repo:app/middleware/session_auth.py",
    "skills/medium-research/SKILL.md::repo:scripts/vault/okf_taxonomy.CANONICAL_TYPES",
    "skills/memory-path-scoping/SKILL.md::repo:scripts/memory/next-gen-memory/context_bundle.py",
    "skills/plan-mode-authoring/SKILL.md::repo:app/middleware/session_auth.py",
    "skills/plan-mode-authoring/SKILL.md::repo:web/src/components/TagChip.tsx",
    "skills/poisoned-worker-troubleshoot/SKILL.md::vault:logs/autonomy_runs/run_",
    "skills/powerpoint/SKILL.md::repo:scripts/office/soffice.py",
    "skills/quick-research/SKILL.md::repo:scripts/vault/okf_taxonomy.CANONICAL_TYPES",
    "skills/subagent-orchestrate/SKILL.md::vault:skills/subagent-orchestrate/subagent_orchestrate.py",
    "skills/system-health-check/SKILL.md::repo:tests/test_health_skill_docs_live_fleet.py",
    "skills/system-health-check/SKILL.md::repo:tests/test_system_health_check_frontend_endpoint.py",
    "skills/system-health-check/SKILL.md::repo:tests/test_system_health_check_skill_fleet.py",
    "skills/voice-clone-sample/SKILL.md::repo:references/ed/ed_001.wav",
    "skills/voice-clone-sample/SKILL.md::repo:references/ed/ed_002.wav",
    "skills/voice-clone-sample/SKILL.md::repo:references/ronan/ronan_001.wav",
    "skills/voice-clone-sample/SKILL.md::repo:scripts/start-cosyvoice-tts.sh",
    "skills/writing-plans/SKILL.md::repo:tests/path/to/test_file.py",
}


def _doc_files() -> list[tuple[str, Path]]:
    """Every active SKILL.md and every autonomy task file, labelled vault-relative."""
    out = [(f"skills/{p.parent.name}/SKILL.md", p)
           for p in sorted((VAULT / "skills").glob("*/SKILL.md"))]
    out += [(f"autonomy/{p.name}", p) for p in sorted((VAULT / "autonomy").glob("*.md"))]
    return out


def _is_template(rel: str) -> bool:
    """A path that could not be opened by anyone, so it cannot be drift."""
    return (any(t in rel for t in ("<", "{", "*", "YYYY", "MM-DD", "..."))
            or rel.endswith(("-", ".", "/")))


def _named_paths(body: str, skill_dir: Path | None) -> set[tuple[str, str]]:
    """(tree, relative path) for each checkout- or vault-rooted path a doc names.

    Three syntactic shapes only, all of them unambiguous about their root: a
    `~/lloyd/…` or `~/obsidian/…` reference anywhere; a backticked
    repo-source-shaped token; and a `.py` path inside a fenced command block.
    A bare relative word (`app/config.py`) is not scanned — it is prose too
    often to be a check, and the whole reason this test is worth writing is
    that the check has no false positives to trade away.
    """
    out: set[tuple[str, str]] = set()
    for m in _LLOYD_PATH.finditer(body):
        if not m.group(1).startswith(RUNTIME_FIRST):
            out.add(("repo", m.group(1)))
    for m in _OBSIDIAN_PATH.finditer(body):
        out.add(("vault", m.group(1)))
    for tok in BACKTICKED.findall(body):
        s = tok.strip()
        if not s:
            continue
        core = re.split(r"[:\s(]", s, maxsplit=1)[0]
        if core.startswith(CHECKOUT_PREFIXES) or core.startswith(SKILL_LOCAL_PREFIXES):
            out.add(("repo", core))
    for block in _command_blocks(body):
        for m in _PY_IN_COMMAND.finditer(block):
            out.add(("repo", m.group(1)))
    return out


def _command_blocks(body: str) -> list[str]:
    """Command blocks a run is meant to copy: fenced, or markdown-indented, and
    only when the block says `~/lloyd` — which is what makes a root claim
    unambiguous instead of a teaching example. The shape task #85 actually used
    was the indented one, so a scanner that read only fences would have passed
    the four phantom nights.
    """
    blocks = [b for b in _FENCE_BLOCK.findall(body) if "~/lloyd" in b]
    cur: list[str] = []
    for line in _FENCE_BLOCK.sub("", body).splitlines():
        if re.match(r"^[ \t]{4,}\S", line) or (cur and not line.strip()):
            cur.append(line)
            continue
        if cur:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _unresolved_rows() -> dict[str, Path]:
    """`<label>::<tree>:<path>` -> the doc file that names it, for every named
    path that is not on disk.

    The doc is kept alongside the row, not thrown away: to say *why* a
    reference is missing, the diagnosis has to ask that file's own repo whether
    the copy on disk is committed (item #1264).
    """
    bad: dict[str, Path] = {}
    for label, path in _doc_files():
        body = path.read_text(encoding="utf-8", errors="replace")
        skill_dir = path.parent if label.startswith("skills/") else None
        for tree, rel in sorted(_named_paths(body, skill_dir)):
            if _is_template(rel) or "/" not in rel or rel.endswith(RUNTIME_SUFFIXES):
                continue
            if tree == "vault":
                roots = [VAULT / rel]
            else:
                roots = [ROOT / rel]
                if skill_dir is not None:
                    roots.append(skill_dir / rel)
            if not any(r.exists() for r in roots):
                bad[f"{label}::{tree}:{rel}"] = path
    return bad


def _unresolved() -> set[str]:
    """`<label>::<tree>:<path>` for every named path that is not on disk."""
    return set(_unresolved_rows())


_GIT_TIMEOUT_S = 20

#: One `git status` per working tree per run, keyed by tree root. Only consulted
#: on a violation, so a green suite never shells out.
_DIRTY_CACHE: dict[str, frozenset[str]] = {}


def _git_status(worktree: str) -> frozenset[str]:
    """Repo-relative paths with uncommitted changes in `worktree` — index,
    worktree, or untracked. Empty if git cannot be asked."""
    if worktree not in _DIRTY_CACHE:
        try:
            proc = subprocess.run(
                ["git", "-C", worktree, "status", "--porcelain"],
                capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            proc = None
        rows: set[str] = set()
        if proc is not None and proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if len(line) < 4:
                    continue
                p = line[3:].strip('"')
                # A rename reports `old -> new`; the new path is the live one.
                if " -> " in p:
                    p = p.split(" -> ", 1)[1]
                rows.add(p)
        _DIRTY_CACHE[worktree] = frozenset(rows)
    return _DIRTY_CACHE[worktree]


def _uncommitted_edit(doc: Path) -> tuple[str, str] | None:
    """(working-tree root, repo-relative path) if `doc` is an uncommitted edit.

    Asked of the file's own tree rather than of `VAULT`, because uncommitted is
    a property of the document's repo and `~/obsidian` is a live tree with no
    worktree of its own. None means committed, outside any repo, or git
    unaskable — the last deliberately falls in with committed, so a diagnosis
    that cannot read git costs the check nothing in leniency (clause 2).
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(doc.parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    top = proc.stdout.strip() if proc.returncode == 0 else ""
    if not top:
        return None
    try:
        rel = str(Path(doc).resolve().relative_to(Path(top).resolve()))
    except ValueError:
        return None
    dirty = _git_status(top)
    if rel in dirty:
        return (top, rel)
    # An untracked *directory* is reported as `dir/`, not file by file, so a new
    # skill's SKILL.md — the likeliest source of a doc-ahead-of-code skew — is
    # under an entry here rather than an entry itself.
    parts = rel.split("/")
    for i in range(1, len(parts)):
        if "/".join(parts[:i]) + "/" in dirty:
            return (top, rel)
    return None


def _drift_report(drift: set[str]) -> str:
    """Say why each unresolved reference is missing.

    Two causes used to print identically, and they need opposite answers. "The
    doc is committed and the file it names never landed" is item #1240's defect
    — put the file in the repo or fix the doc. "The doc naming it is an
    uncommitted edit in the vault working tree" is item #1264's: the instruction
    was written ahead of anything committed, so the code gap may not exist at
    all, and the vault edit is the thing to commit or revert. A red node in
    `~/lloyd` cannot be attributed to or reverted by a commit if the second case
    is reported as the first, which is how four rounds on 2026-09-19 were refused
    for a skew no commit had produced.

    This function only *explains*. The assertion it feeds still requires an
    empty drift either way — the diagnosis adds no leniency.
    """
    rows = _unresolved_rows()
    gaps: list[str] = []
    skews: list[tuple[str, str, str]] = []
    for row in sorted(drift):
        doc = rows.get(row)
        un = _uncommitted_edit(doc) if doc is not None else None
        if un is None:
            gaps.append(row)
        else:
            skews.append((row, un[1], un[0]))
    parts: list[str] = []
    if gaps:
        parts.append(
            "skills or autonomy tasks name paths that are not in this checkout: "
            f"{gaps}. Either put the file in the repo, point the doc at where it "
            "really lives, or — only for pre-existing drift this round did not "
            "touch — add it to PATH_KNOWN_UNFIXED with a reason.")
    for row, rel, top in skews:
        # What the doc names, pulled out of the row itself (`label::tree:path`),
        # so the command a reader pastes has no placeholder left in it.
        named = row.split("::", 1)[1].split(":", 1)[1]
        parts.append(
            f"{row}: the naming file {rel} is an UNCOMMITTED EDIT in the working "
            f"tree at {top} — a working-tree skew, so this row is not evidence "
            "that the code never landed. The path may be named only by text no "
            "commit has yet. Ask the committed copy first: "
            f"`git -C {top} show HEAD:{rel} | grep -c '{named}'` — and if that "
            "says 0, committing or reverting that one vault file is the fix, not "
            "adding the path to PATH_KNOWN_UNFIXED.")
    return "\n".join(parts)


def test_no_active_skill_or_task_names_a_path_absent_from_the_checkout():
    """Task #85 "succeeded" four nights running while its named script was not
    in the tree: the run found the filename in its instructions, could not open
    it, improvised a scratch `git worktree` of an unmerged branch, measured an
    `app/` three days older than production, and reported numbers. Nothing in
    the suite could see that, because no test asked whether a path a doc names
    exists (item #1240).

    Deliberately unmarked — not `live_vault` — like the phantom-tool test above
    it mirrors. That is the point of the ledger: without it, this assertion
    would block unrelated rounds for prose they did not write, and a check that
    blocks the wrong thing gets disabled.

    The message is `_drift_report`, because on 2026-09-19 four rounds were
    refused by a violation that no commit had produced: a skill edited in the
    vault working tree and never committed, naming a script still on an
    unmerged branch. Same red, opposite fix, and nothing in the text
    distinguished them (item #1264).
    """
    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == set(), _drift_report(drift)


def test_the_path_check_anchored_to_a_path_that_really_exists():
    """Positive control. A scanner whose patterns all miss would report zero
    violations and read exactly like a clean board — the failure mode this whole
    item is about, so the check has to prove it can resolve something before its
    empty result means anything."""
    body = (VAULT / "skills" / "secondary-routing-eval" / "SKILL.md").read_text(
        encoding="utf-8", errors="replace")
    refs = {rel for _tree, rel in _named_paths(body, VAULT / "skills" / "secondary-routing-eval")}
    assert "eval/secondary_routing_eval.py" in refs, (
        f"the scanner stopped resolving the script the eval skill names; got {sorted(refs)}")
    assert (ROOT / "eval" / "secondary_routing_eval.py").exists(), (
        "and the script it resolved is itself missing from the checkout")


def test_the_path_check_goes_red_on_a_reference_it_should_see(tmp_path, monkeypatch):
    """Negative control, and the one that matters most here: the four nights of
    task #85 were invisible because the only check that existed matched a
    filename *inside* a task file and never asked whether the file it named was
    present. So the control is exactly that shape — a task whose Step 1 names a
    script that is not in the checkout — and if the scanner cannot catch it, the
    empty result of the real assertion above means nothing."""
    real = _doc_files()
    decoy = tmp_path / "autonomy" / "9999-fake.md"
    decoy.parent.mkdir()
    decoy.write_text(
        "---\nname: fake\ntype: autonomy\n---\n"
        "# Fake Task\n\n    cd ~/lloyd && .venvs/lloyd/bin/python "
        "eval/a_script_that_does_not_exist_1240.py --repeats 3\n",
        encoding="utf-8")
    import sys
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: real + [("autonomy/9999-fake.md", decoy)])
    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {"autonomy/9999-fake.md::repo:eval/a_script_that_does_not_exist_1240.py"}, (
        f"the scanner did not isolate the one planted violation: {sorted(drift)}")


#: The script both controls below name. It is not in the checkout and never will be.
_SKEW_SCRIPT = "scripts/a_script_only_a_vault_edit_names_1264.py"


def _task_body(*, names_script: bool) -> str:
    """An autonomy task file, optionally naming a script that is not in the checkout.

    The indented command block is the shape task #85 actually used, and the
    `~/lloyd` inside it is what makes the root claim unambiguous to the scanner.
    """
    head = "---\nname: skew\ntype: autonomy\n---\n# Skew Task\n\nStep 1.\n"
    if not names_script:
        return head + "\nNothing runnable here yet.\n"
    return head + ("\n    cd ~/lloyd && .venvs/lloyd/bin/python "
                   f"{_SKEW_SCRIPT} --window 30\n")


def _git_repo_with_committed_task(tmp_path: Path) -> Path:
    """A fresh git working tree standing in for `~/obsidian`, holding one clean
    committed autonomy task. Returns the tree root; it is clean on return.

    A real repo, not a mocked dirty-set: "is this file committed?" is answered by
    a `git status` subprocess, so the control has to cross that boundary to prove
    anything about the answer.
    """
    repo = tmp_path / "obsidian"
    (repo / "autonomy").mkdir(parents=True)
    (repo / "autonomy" / "9999-skew.md").write_text(
        _task_body(names_script=False), encoding="utf-8")
    for cmd in (["init", "-q"],
                # A machine's global excludesfile must not decide the fixture.
                ["-c", "core.excludesfile=/dev/null", "add", "autonomy/9999-skew.md"],
                ["-c", "user.name=lloyd-test", "-c", "user.email=lloyd-test@invalid",
                 "commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=str(repo), check=True,
                       capture_output=True, text=True)
    return repo


def test_an_uncommitted_vault_edit_is_diagnosed_as_a_skew_not_a_missing_file(tmp_path,
                                                                            monkeypatch):
    """Clause 1 (#1264). The exact state that made main red on 2026-09-19: a task
    committed with no such instruction, then edited in the working tree to name a
    script nothing has committed. The failure text must call that an uncommitted
    edit and name the file — and must NOT also file the row under "not in this
    checkout", because "the code never landed" is the conclusion that sent four
    rounds to their deaths.

    The decoy is planted alone rather than among the live corpus: the assertion
    here is about which sentence the row is described by, and a stray
    real-corpus violation would describe a *different* row, not this one.
    """
    repo = _git_repo_with_committed_task(tmp_path)
    doc = repo / "autonomy" / "9999-skew.md"
    doc.write_text(_task_body(names_script=True), encoding="utf-8")   # left uncommitted

    label = "autonomy/9999-skew.md"
    monkeypatch.setattr(sys.modules[__name__], "_doc_files", lambda: [(label, doc)])
    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"{label}::repo:{_SKEW_SCRIPT}"}, (
        f"the planted uncommitted-edit violation was not seen: {sorted(drift)}")

    msg = _drift_report(drift)
    assert "UNCOMMITTED EDIT" in msg, f"the text never says the file is uncommitted: {msg}"
    assert label in msg, f"the text never names the uncommitted file: {msg}"
    assert str(repo) in msg, f"the text never says which working tree: {msg}"
    assert "working-tree skew" in msg, f"the text never names the diagnosis: {msg}"
    assert "not in this checkout" not in msg, (
        f"a working-tree skew was still reported as a missing file: {msg}")


def test_a_committed_doc_naming_an_absent_file_still_goes_red_alone(tmp_path, monkeypatch):
    """Clause 2 (#1264): the same fixture one bit away from the control above —
    here the instruction IS committed and the tree is clean — and the row is still
    the sole isolated violation, reported as missing from the checkout, with no
    mention of an uncommitted edit. This is what stops the diagnosis becoming a
    leniency: `_uncommitted_edit` returning None is the committed case, the
    not-a-repo case and the git-unaskable case at once, and all three keep the
    teeth #1240 gave the check.
    """
    repo = _git_repo_with_committed_task(tmp_path)
    doc = repo / "autonomy" / "9999-skew.md"
    doc.write_text(_task_body(names_script=True), encoding="utf-8")
    for cmd in (["add", "autonomy/9999-skew.md"],
                ["-c", "user.name=lloyd-test", "-c", "user.email=lloyd-test@invalid",
                 "commit", "-q", "-m", "the instruction"]):
        subprocess.run(["git", *cmd], cwd=str(repo), check=True,
                       capture_output=True, text=True)

    label = "autonomy/9999-skew.md"
    real = _doc_files()
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: real + [(label, doc)])
    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"{label}::repo:{_SKEW_SCRIPT}"}, (
        f"a committed doc naming an absent file stopped being caught: {sorted(drift)}")

    msg = _drift_report(drift)
    assert "not in this checkout" in msg, f"the row lost its real diagnosis: {msg}"
    assert "UNCOMMITTED EDIT" not in msg, (
        f"a committed doc was excused as a working-tree skew: {msg}")


def test_the_secondary_routing_nightly_docs_are_clean():
    """The two docs this round rewrote must resolve with no help from the
    ledger: the script, the item set, the decision artifact, the report the run
    reads, and the trend note it appends to must all be findable, one location
    each."""
    drift = {e for e in _unresolved()
             if e.startswith("skills/secondary-routing-eval/") or e.startswith("autonomy/85-")}
    assert drift == set(), f"the routing eval's own docs are still broken: {sorted(drift)}"


def _routing_docs() -> dict[str, str]:
    """The two documents that tell the nightly run what to do, keyed by label.

    The skill and the autonomy task are the same run's instructions read in
    either order, so a sentence corrected in one and not the other is a
    sentence the run will follow differently depending on which it opened.
    """
    return {
        "skill": (VAULT / "skills" / "secondary-routing-eval" / "SKILL.md").read_text(
            encoding="utf-8", errors="replace"),
        "task": next(iter((VAULT / "autonomy").glob("85-*.md"))).read_text(
            encoding="utf-8", errors="replace"),
    }


def test_the_routing_skill_states_the_real_per_job_route():
    """Step 4 used to say a flip is made "in `config.yaml`" and that "a per-job
    route … does not exist yet". Both were false the moment
    `JOBS_ON_PRIMARY`/`_engine_for` landed: `config.yaml` has exactly two model
    slots, `primary` and `secondary`, and no per-job key, and following that
    sentence would have had a nightly set `secondary_enabled: false` — taking
    the engine down for all four jobs that were measured as keeps.

    Both docs are held to it, not just the skill: the task file repeats the
    procedure for the run that reads the task row instead, and it named no
    route at all until item #1240, which left "how do I apply a flip"
    unanswered exactly where a flip is what the nightly is looking for.
    """
    for label, body in _routing_docs().items():
        assert "JOBS_ON_PRIMARY" in body, (
            f"the {label} must name the knob that routes a job "
            "(app/secondary_models.py, consulted by _engine_for)")
        assert "--pin" in body, (
            f"the {label} must name the command that writes it; the module "
            "constant is never edited by hand")
        assert "not a `config.yaml` setting" in body, (
            f"the {label} must say where the route does NOT live: the only "
            "per-engine key in config.yaml is secondary_enabled, which starts "
            "and stops the engine for every job at once")
        assert "which does not exist yet" not in body, (
            f"the {label} again claims the per-job route is missing; it landed "
            "in app/secondary_models.py")


def test_the_two_nightly_docs_name_one_location_for_the_trend():
    """The skill and the task are the same run's instructions, read in either
    order. They disagreed before: the skill said append to
    `eval/secondary-routing/trend.md`, the task said the same, and the four real
    runs wrote the vault instead — so whichever path this round picked, both
    docs had to move together or the next night drifted again."""
    skill = (VAULT / "skills" / "secondary-routing-eval" / "SKILL.md").read_text(
        encoding="utf-8", errors="replace")
    task = next(iter((VAULT / "autonomy").glob("85-*.md"))).read_text(
        encoding="utf-8", errors="replace")
    for doc, name in ((skill, "skill"), (task, "task")):
        assert "projects/lloyd/secondary-routing-eval/trend.md" in doc, (
            f"the {name} must point the trend row at the vault note that holds it")
        assert "eval/secondary-routing/trend.md" not in doc, (
            f"the {name} still names a repo trend file that nothing writes")

