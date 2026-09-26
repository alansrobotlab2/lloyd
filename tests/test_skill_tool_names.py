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
    that `Agent` was never one of them. Two active skills still name it —
    `strict-task-mapping` instructs the call (#410), and `discord-social`
    forbids it, which is correct prose that any matcher has to survive. #409
    moved `deep-research` and `research-agent` onto `Task` and archived
    `subagent-orchestrate`. Because `Agent` is
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

# Bound here, not read at call time, so a test can point the fence at a fixture.
from app.paths import IS_WORKTREE, LIVE_CHECKOUT  # noqa: E402

SKILLS_DIRS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]

# Tool names that were never real in Lloyd, or no longer are. The first group
# was found in active skills on 2026-09-04.
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
    # Retired on 2026-09-23 as duplicates or subsumed. The four #376 verbs
    # (remember, recall, forget, improve) went too but are English words, so a
    # word-boundary match on them would flag ordinary prose.
    "fact_profile", "fact_check", "browser_type",
    "autoresearch_round", "autoresearch_promote", "autoresearch_bench_add",
    "autoresearch_bench_list", "autoresearch_ledger_query",
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

#: The marker left where something was cut short. `clip_skill_description` says so
#: of itself — "cut to at most `max_chars` characters, the cut marked with `…`"
#: (`prompt_builder.py:992-996`) — and `scripts/skill_lint.py:822` puts that clipped
#: string into a table cell of the report it emits, which is a document this guard
#: scans. `...` is the same event typed by a human.
CLIP_MARKERS = ("\N{HORIZONTAL ELLIPSIS}", "...")


def _truncated(cand: str, tail: str = "") -> bool:
    """True when `cand` is a quotation that ran out of room, not a path being named.

    Two shapes, one rule, because the marker lands on a different side per matcher:
    inside the candidate (the backticked class allows any non-backtick character, so
    it carries the marker through) or immediately after it (the two anchored classes
    stop at it, so the report's cell reads `…~/lloyd/agent-services/su…` and yields
    `agent-services/su`). Either way the text is not asserting that a file named
    `agent-services/su` should exist — it is quoting a real name that continues past
    what the budget allowed. A reference this guard cannot complete is not checkable
    against the checkout, so the honest verdict is "not a claim", the same verdict
    `_is_template` reaches for a `<slug>`; the alternative is a red node that no
    change to the code can clear.
    """
    return any(m in cand for m in CLIP_MARKERS) or tail.startswith(CLIP_MARKERS)

#: Every unresolved reference that existed on 2026-09-18, when this check was
#: written, across 230 skill/task files and 381 references. Same contract as
#: `KNOWN_UNFIXED` above: real drift, each one worth fixing, held out of the
#: assertion so an unrelated round is not blocked by prose it did not touch. A
#: NEW entry here is a regression; `autonomy/85-*` and `secondary-routing-eval`
#: are deliberately NOT in it — they are what item #1240 fixed, and
#: `test_the_secondary_routing_nightly_docs_are_clean` keeps them out.
PATH_KNOWN_UNFIXED: set[str] = {
    "skills/ai-engineer-monitor/SKILL.md::vault:autonomy/75-ai-engineer-youtube-monitor.md",
    "skills/deep-research/SKILL.md::repo:scripts/vault/okf_taxonomy.CANONICAL_TYPES",
    "skills/documentation-digester/SKILL.md::repo:agent-services/llm/llama.cpp",
    "skills/documentation-digester/SKILL.md::repo:agent-services/llm/llama.cpp/build/bin/llama-server",
    "skills/entity-resolution-sweep/SKILL.md::repo:agent_mcp/memory.py",
    "skills/file-path-resolution/SKILL.md::repo:inner-voice/system_prompt.md",
    "skills/file-path-resolution/SKILL.md::repo:lloyd/inner-voice/system-prompt.md",
    "skills/file-path-resolution/SKILL.md::vault:lloyd/inner-voice/system-prompt.md",
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
        if not m.group(1).startswith(RUNTIME_FIRST) and not _truncated(m.group(1),
                                                                      body[m.end():]):
            out.add(("repo", m.group(1)))
    for m in _OBSIDIAN_PATH.finditer(body):
        if not _truncated(m.group(1), body[m.end():]):
            out.add(("vault", m.group(1)))
    for tok in BACKTICKED.findall(body):
        s = tok.strip()
        if not s:
            continue
        core = re.split(r"[:\s(]", s, maxsplit=1)[0]
        if _truncated(core):
            continue
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


#: One batched `git check-ignore` per scan. Like `_DIRTY_CACHE` this is reached
#: only once something is already missing, so a green suite still never shells out.
_IGNORE_QUERY_TIMEOUT_S = 20


def _tree_ignores(root: Path, relpaths) -> frozenset[str]:
    """The subset of `relpaths` that `root`'s OWN ignore rules exclude.

    Asked of git in one batched `--stdin` call rather than from a list kept here,
    which is what makes this a rule and not an enumeration: the answer is the
    tree's `.gitignore` plus `.git/info/exclude` plus any `core.excludesFile`, so
    the next path someone puts into a skill is classified by the same mechanism
    that classified the last one. `--no-index` because the question is "do the
    tree's ignore rules cover this path?", not "is this path tracked" — and the
    path is absent here by definition, since only absent paths reach this point.

    Every failure mode is fail-closed to `frozenset()`: git unaskable (not a
    repo, timeout, signal) exempts nothing and each absent reference stays
    reported. An exemption that widened when its own oracle could not be asked
    would be a guard reading its own missing input, the defect `lloyd/MEMORY.md`
    keeps a catalogue of.
    """
    paths = sorted({p for p in relpaths if p})
    if not paths:
        return frozenset()
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"],
            input="\n".join(paths) + "\n",
            capture_output=True, text=True, timeout=_IGNORE_QUERY_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    # 0 = at least one path is ignored, 1 = none is. Anything else is git failing
    # to answer, which is the unaskable case above.
    if proc.returncode not in (0, 1):
        return frozenset()
    return frozenset(line.strip() for line in proc.stdout.splitlines() if line.strip())


def _present_at(repo: Path, rev: str, relpaths: list[str]) -> frozenset[str]:
    """The subset of `relpaths` that commit `rev` of `repo` carries, from one
    batched `cat-file`. Raises on a git that cannot answer; callers fail closed."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch-check=%(objecttype)"],
        input="".join(f"{rev}:{p}\n" for p in relpaths),
        capture_output=True, text=True, timeout=_IGNORE_QUERY_TIMEOUT_S, check=True)
    lines = proc.stdout.splitlines()
    if len(lines) != len(relpaths):
        raise subprocess.SubprocessError("cat-file answered a different number of lines")
    return frozenset(p for p, line in zip(relpaths, lines) if not line.endswith(" missing"))


def _landed_after_base(root: Path, live: Path, relpaths) -> frozenset[str]:
    """The subset of `relpaths` that `live`'s HEAD carries and the merge-base of
    `root`'s HEAD with it does not: files that landed on main AFTER this
    worktree's base, which a rebase brings in and nothing in the round can fix.

    The merge-base is what keeps this an exemption for the base and not for the
    branch: a file the round itself deleted is at the base and at live HEAD
    both, so it stays a violation. Every failure — git unaskable, the two trees
    sharing no history, a timeout — is fail-closed to `frozenset()`, the rule
    `_tree_ignores` follows and for the same reason.
    """
    paths = sorted({p for p in relpaths if p})
    if not paths:
        return frozenset()
    try:
        live_head = subprocess.run(
            ["git", "-C", str(live), "rev-parse", "--verify", "HEAD"],
            capture_output=True, text=True, timeout=_IGNORE_QUERY_TIMEOUT_S,
            check=True).stdout.strip()
        base = subprocess.run(
            ["git", "-C", str(root), "merge-base", "HEAD", live_head],
            capture_output=True, text=True, timeout=_IGNORE_QUERY_TIMEOUT_S,
            check=True).stdout.strip()
        if not live_head or not base:
            return frozenset()
        return _present_at(live, live_head, paths) - _present_at(root, base, paths)
    except (OSError, subprocess.SubprocessError):
        return frozenset()


#: `<label>::repo:<path>` -> naming doc, for every reference the last scan set
#: aside as out of scope: at live HEAD, not at this worktree's base. Filled by
#: `_unresolved_rows` and read by `_drift_report`, so the tally says what was
#: dropped and why rather than dropping it silently (#1411).
_OUT_OF_SCOPE: dict[str, Path] = {}


def _row_parts(row: str) -> tuple[str, str]:
    """(tree, rel) out of a `<label>::<tree>:<rel>` row — the same split
    `_drift_report` uses to pull out the path a doc names."""
    tree, rel = row.split("::", 1)[1].split(":", 1)
    return tree, rel


def _unresolved_rows() -> dict[str, Path]:
    """`<label>::<tree>:<path>` -> the doc file that names it, for every named
    path that is not on disk.

    The doc is kept alongside the row, not thrown away: to say *why* a
    reference is missing, the diagnosis has to ask that file's own repo whether
    the copy on disk is committed (item #1264).

    A `repo:` reference the tree itself ignores is dropped before the comparison
    (item #1403), because such a path is not checkable from a worktree at all: a
    round's tree is `git worktree add` from HEAD, which by construction holds no
    ignored file. `qmd/dist/cli/qmd.js` — present in the live checkout, named by
    `skills/qmd-index-maintenance/SKILL.md` since vault commit `3d0738d1`, and
    ignored by `.gitignore` `/qmd/` — was therefore red in every round and green
    in `~/lloyd`, three nodes, since the two negative controls assert their
    planted violation is the SOLE drift and that row rode along with theirs.
    `RUNTIME_SUFFIXES` above already exempts by *suffix* for the same reason ("a
    file on the machine, not drift"); this exempts by *ignore rule*, which needs
    no entry per doc and so cannot be re-broken by one prose commit. `vault:`
    rows are never exempted here: their tree is the vault, whose ignore rules are
    not the ones this asks.

    The corpus is the LIVE vault while the tree is whatever checkout this file
    runs from, and the two are only the same age in `~/lloyd` (item #1411). A
    skill that names `architecture/desktop.md` the day it lands is correct
    against `main` and a dangling reference in every open round whose base
    predates that commit — 213 refused gate rows on 2026-09-23 alone, and the
    base probe faithfully reproduces it, because the skew is a property of the
    checkout and not of the diff. So in a worktree (`IS_WORKTREE`) a `repo:`
    reference that live HEAD carries and this worktree's base does not is set
    aside into `_OUT_OF_SCOPE` rather than counted: a rebase resolves it and no
    edit here can. `_landed_after_base` says why the merge-base, not the base
    itself, is the second side of that comparison. Never applied outside a
    worktree, where the tree IS what the vault was written against.
    """
    _OUT_OF_SCOPE.clear()
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
    ignored = _tree_ignores(ROOT, [_row_parts(r)[1] for r in bad
                                   if _row_parts(r)[0] == "repo"])
    if ignored:
        bad = {r: d for r, d in bad.items()
               if not (_row_parts(r)[0] == "repo" and _row_parts(r)[1] in ignored)}
    if IS_WORKTREE:
        landed = _landed_after_base(ROOT, LIVE_CHECKOUT,
                                    [_row_parts(r)[1] for r in bad
                                     if _row_parts(r)[0] == "repo"])
        for r in list(bad):
            if _row_parts(r)[0] == "repo" and _row_parts(r)[1] in landed:
                _OUT_OF_SCOPE[r] = bad.pop(r)
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


def _tracked_at_head(top: str, rel: str) -> bool | None:
    """Whether `top`'s HEAD holds `rel` at all. None when git cannot say, which
    the caller words as the tracked case — the hint that costs least if wrong."""
    try:
        return rel in _present_at(Path(top), "HEAD", [rel])
    except (OSError, subprocess.SubprocessError):
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

    A third case is the uncommitted edit's limit (#1411): a naming file that is
    NEW — untracked, in no commit at all, the shape of a skill written today.
    The tracked-file hint (`git show HEAD:<rel> | grep -c`) necessarily says 0
    for it and then offers two remedies that both fail: committing keeps the
    citation and stays red (`test_a_committed_doc_naming_an_absent_file_still_
    goes_red_alone` is what pins that), reverting deletes a skill somebody
    wrote. So that case gets its own sentence, pointing at the cited path.

    This function only *explains*. The assertion it feeds still requires an
    empty drift either way — the diagnosis adds no leniency. The one thing it
    adds is the tally of rows the scan set aside as out of scope, so a red
    report in a worktree also says which references it deliberately did not
    count.
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
        if _tracked_at_head(top, rel) is False:
            parts.append(
                f"{row}: the naming file {rel} is NEW and UNTRACKED in the working "
                f"tree at {top} — a working-tree skew with no committed copy to "
                "compare against, so this row is not evidence that the code never "
                "landed. Committing that file keeps the citation and stays red; "
                "reverting it deletes a file someone wrote. The fix is on the cited "
                f"side: put {named} in the repo or point the doc at where it really "
                "lives — and if it is already at live HEAD, a rebase clears this row.")
            continue
        parts.append(
            f"{row}: the naming file {rel} is an UNCOMMITTED EDIT in the working "
            f"tree at {top} — a working-tree skew, so this row is not evidence "
            "that the code never landed. The path may be named only by text no "
            "commit has yet. Ask the committed copy first: "
            f"`git -C {top} show HEAD:{rel} | grep -c '{named}'` — and if that "
            "says 0, committing or reverting that one vault file is the fix, not "
            "adding the path to PATH_KNOWN_UNFIXED.")
    if _OUT_OF_SCOPE:
        parts.append(
            f"OUT OF SCOPE for this worktree, not counted: {sorted(_OUT_OF_SCOPE)} "
            f"— each names a file at live HEAD ({LIVE_CHECKOUT}) that this "
            "checkout's base predates. A rebase resolves them; nothing here needs "
            "fixing for them.")
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


def test_an_arrow_corrected_citation_still_names_the_phantom(tmp_path, monkeypatch):
    """Why #1454's fix had to be the document and not the check.

    The row that turned three nodes red was an autonomy brief recording a
    corrected citation as ``eval/djev/skills.py``→``eval/djev/schemas.py:315-325``.
    The right-hand name is real; the left-hand one has never existed in the
    checkout (``git log --all -- eval/djev/skills.py`` is empty), and the writer
    believed the arrow retired it. It does not: ``BACKTICKED`` takes every
    backticked token and ``core`` cuts at the first ``:``, so a phantom wrapped in
    a correction is scanned exactly like a live citation — which is the behaviour
    this control pins, on the same tree that the real row was reported against.

    The alternative fix was to exempt the arrow form, and that is the hole this
    closes: a doc that merely *looks* self-correcting would then carry a
    never-existing path into every run that reads it, which is precisely the
    defect class #1240 opened. The remedy that stays correct is the one taken —
    keep the true path, drop the phantom (vault commit ``bc3bd1fe``).
    """
    decoy = tmp_path / "autonomy" / "9998-arrow.md"
    decoy.parent.mkdir()
    decoy.write_text(
        "---\nname: arrow\ntype: autonomy\n---\n"
        "# Arrow Task\n\nLive name sets: "
        "`eval/djev/a_phantom_before_1454.py`→`eval/djev/schemas.py:315-325`.\n",
        encoding="utf-8")
    # Planted ALONE, not appended to the live corpus: the assertion is an exact set,
    # so a corpus-wide scan would let unrelated live-vault drift — a doc some other
    # job is mid-editing — redden this node for a reason that has nothing to do with
    # the arrow form. Same reason the skew control below replaces the corpus outright.
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: [("autonomy/9998-arrow.md", decoy)])
    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {"autonomy/9998-arrow.md::repo:eval/djev/a_phantom_before_1454.py"}, (
        f"the corrected-citation form was not isolated to its phantom half: {sorted(drift)}")


#: The two names the ignore-rule control below plants, one under each tree.
_IGNORED_TREE = "ignored_by_the_tree"
_TRACKED_TREE = "tracked_source"
_PLANTED_SCRIPT = "a_script_no_commit_has_1403.py"


def _fixture_checkout_with_its_own_ignore_rules(tmp_path: Path) -> Path:
    """A fresh git working tree standing in for a checkout, whose `.gitignore`
    ignores exactly one top-level directory. Returns the tree root.

    A real `git init`, for the reason `_git_repo_with_committed_task` gives: the
    exemption is answered by a `git check-ignore` subprocess over the tree's own
    rules, so the control has to cross that boundary to prove the answer came
    from a tree and not from a name this file recognises. `.gitignore` is written
    into the FIXTURE, never into the live checkout — `#1403` clause 2 forbids a
    round from writing the real `.gitignore`, and the gate denies it anyway.
    """
    repo = tmp_path / "checkout"
    (repo / _IGNORED_TREE).mkdir(parents=True)
    (repo / _TRACKED_TREE).mkdir()
    (repo / ".gitignore").write_text(f"/{_IGNORED_TREE}/\n", encoding="utf-8")
    (repo / _TRACKED_TREE / "keep.py").write_text("x = 1\n", encoding="utf-8")
    for cmd in (["init", "-q"],
                # A machine's global excludesfile must not decide the fixture.
                ["-c", "core.excludesfile=/dev/null", "add", "."],
                ["-c", "user.name=lloyd-test", "-c", "user.email=lloyd-test@invalid",
                 "commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=str(repo), check=True,
                       capture_output=True, text=True)
    return repo


def test_a_ref_under_an_ignored_tree_is_dropped_and_a_tracked_one_is_not(
        tmp_path, monkeypatch):
    """#1403 clause 1, both halves in ONE scan of one fixture tree.

    The exemption must not become a second way to be blind, so the dropped row
    and the still-reported row are planted in the same document and read by the
    same call: an ignored first component drops the reference, a tracked one
    leaves it exactly as chargeable as the control above. `_uncommitted_edit` is
    irrelevant here — the assertion is about which rows exist at all, and it is
    `==`, so a fixture that leaked anything else fails.

    `ROOT` is pointed at the fixture so the exemption is computed from a tree the
    test wrote, which is the only way to show the answer comes from the tree's
    ignore rules rather than from `/qmd/` being hard-coded here.
    """
    repo = _fixture_checkout_with_its_own_ignore_rules(tmp_path)
    doc = tmp_path / "9999-ignore.md"
    doc.write_text(
        "---\nname: ignore\ntype: autonomy\n---\n# Ignore Task\n\n"
        "Step 1 names a script under a tree the checkout ignores: "
        f"`~/lloyd/{_IGNORED_TREE}/{_PLANTED_SCRIPT}`.\n\n"
        "Step 2 names one under a tree it does not: "
        f"`~/lloyd/{_TRACKED_TREE}/{_PLANTED_SCRIPT}`.\n",
        encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", repo)
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: [("autonomy/9999-ignore.md", doc)])
    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"autonomy/9999-ignore.md::repo:{_TRACKED_TREE}/{_PLANTED_SCRIPT}"}, (
        f"the ignore rule did not drop exactly the ignored-tree ref and only that ref: "
        f"{sorted(drift)}")


def test_the_qmd_row_is_dropped_by_the_rule_and_not_by_a_ledger_entry():
    """#1403 clause 2's teeth, stated against the real tree.

    The green of the three path nodes has two possible causes and only one of
    them is the fix: the exemption covering `qmd/dist/cli/qmd.js`, or the scanner
    losing the reference. So this asks for the reference from the doc that put it
    there (vault commit `3d0738d1`, 2026-09-23, one prose commit that red three
    nodes for every round), from git's own answer, and from the ledger — which
    must stay empty of it. `PATH_KNOWN_UNFIXED` is a hand-maintained set keyed by
    doc; #1317 closed on that shape and the property re-broke in twelve hours, so
    an entry here would not be a smaller fix, it would be the old bug re-armed.
    """
    skill_dir = VAULT / "skills" / "qmd-index-maintenance"
    refs = {rel for _tree, rel in _named_paths(
        (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace"),
        skill_dir)}
    assert "qmd/dist/cli/qmd.js" in refs, (
        f"the scanner no longer resolves the path that skill names, so the green of "
        f"the path nodes proves nothing; the doc may have changed — got {sorted(refs)}")

    # The tree's rules, not this file's: the one real row is ignored, and the
    # script-shaped path the planted controls use is not.
    assert _tree_ignores(ROOT, ["qmd/dist/cli/qmd.js",
                                "eval/a_script_that_does_not_exist_1240.py"]) \
        == frozenset({"qmd/dist/cli/qmd.js"}), (
        "git's answer for these two paths is not the one the exemption depends on — "
        "either `.gitignore` stopped ignoring /qmd/ (then the row is real drift again) "
        "or the query stopped working")

    enumerated = sorted(e for e in PATH_KNOWN_UNFIXED if "qmd" in e)
    assert not enumerated, (
        f"the exemption was enumerated instead of derived: {enumerated} — item #1403 "
        "clause 2 is that no `skills/qmd-index-maintenance/SKILL.md::repo:qmd/...` "
        "row goes in here")


#: The three paths the fence control below plants: one that landed on the live
#: tree after the worktree's base, one only the live WORKING tree has, one in
#: neither tree.
_LANDED_AFTER_BASE = "eval/a_script_that_landed_after_the_base_1411.py"
_UNCOMMITTED_AT_LIVE = "eval/a_script_only_the_live_working_tree_has_1411.py"
_ABSENT_EVERYWHERE = "eval/a_script_no_tree_has_1411.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=lloyd-test", "-c", "user.email=lloyd-test@invalid",
                    "-c", "core.excludesfile=/dev/null", *args],
                   cwd=str(repo), check=True, capture_output=True, text=True)


def _live_tree_and_an_older_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A stand-in for `~/lloyd` and a `git worktree` of it cut one commit
    earlier — the shape of every open round the moment a doc lands on main.

    Returns `(live, worktree)`. `live` has two commits: a base carrying
    `tracked_source/keep.py`, and one more adding `_LANDED_AFTER_BASE`. It also
    holds `_UNCOMMITTED_AT_LIVE` in its working tree only. The worktree sits at
    the base. Real git on both sides, because the fence is answered by
    `merge-base` and `cat-file` subprocesses over the two trees' shared object
    store, and the control has to prove the answer came from there.
    """
    live = tmp_path / "live"
    (live / _TRACKED_TREE).mkdir(parents=True)
    (live / _TRACKED_TREE / "keep.py").write_text("x = 1\n", encoding="utf-8")
    _git(live, "init", "-q")
    _git(live, "add", ".")
    _git(live, "commit", "-q", "-m", "base")
    wt = tmp_path / "round"
    _git(live, "worktree", "add", "-q", "--detach", str(wt), "HEAD")
    (live / _LANDED_AFTER_BASE).parent.mkdir()
    (live / _LANDED_AFTER_BASE).write_text("y = 2\n", encoding="utf-8")
    _git(live, "add", _LANDED_AFTER_BASE)
    _git(live, "commit", "-q", "-m", "landed after the base")
    (live / _UNCOMMITTED_AT_LIVE).write_text("z = 3\n", encoding="utf-8")
    assert not (wt / _LANDED_AFTER_BASE).exists()
    return live, wt


def _doc_naming(tmp_path: Path, *rels: str) -> Path:
    doc = tmp_path / "9999-fence.md"
    doc.write_text("---\nname: fence\ntype: autonomy\n---\n# Fence Task\n\n"
                   + "".join(f"Step names `~/lloyd/{r}`.\n\n" for r in rels),
                   encoding="utf-8")
    return doc


def _point_the_fence(monkeypatch, *, root: Path, live: Path, worktree: bool) -> None:
    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod, "LIVE_CHECKOUT", live)
    monkeypatch.setattr(mod, "IS_WORKTREE", worktree)


def test_a_ref_that_landed_after_the_base_is_out_of_scope_in_a_worktree(tmp_path,
                                                                        monkeypatch):
    """#1411, the mechanism itself. Three references in one doc, one scan: the
    file that landed on the live tree after this worktree's base is set aside
    and named in the tally; the file only the live WORKING tree holds is still
    a violation (HEAD is the question, a rebase brings in commits); the file no
    tree has is still a violation. `==`, so a fence that dropped more than the
    one row it should fails here."""
    live, wt = _live_tree_and_an_older_worktree(tmp_path)
    doc = _doc_naming(tmp_path, _LANDED_AFTER_BASE, _UNCOMMITTED_AT_LIVE, _ABSENT_EVERYWHERE)
    _point_the_fence(monkeypatch, root=wt, live=live, worktree=True)
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: [("autonomy/9999-fence.md", doc)])

    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"autonomy/9999-fence.md::repo:{_UNCOMMITTED_AT_LIVE}",
                     f"autonomy/9999-fence.md::repo:{_ABSENT_EVERYWHERE}"}, (
        f"the fence did not set aside exactly the landed-after-base row: {sorted(drift)}")
    assert set(_OUT_OF_SCOPE) == {f"autonomy/9999-fence.md::repo:{_LANDED_AFTER_BASE}"}

    msg = _drift_report(drift)
    assert "OUT OF SCOPE" in msg and _LANDED_AFTER_BASE in msg, (
        f"the report never says which row it set aside, or why: {msg}")
    assert str(live) in msg, f"the report never names the live tree it compared against: {msg}"


def test_a_file_the_round_itself_removed_is_still_a_violation(tmp_path, monkeypatch):
    """The leniency the merge-base exists to refuse. `keep.py` is at the base
    and at live HEAD; the worktree deletes it. "Present at live HEAD, absent
    here" is true of it exactly as of a file that landed after the base, and
    only the base tells them apart — a doc naming a file this branch removed is
    the drift #1240 was written for."""
    live, wt = _live_tree_and_an_older_worktree(tmp_path)
    (wt / _TRACKED_TREE / "keep.py").unlink()
    doc = _doc_naming(tmp_path, f"{_TRACKED_TREE}/keep.py")
    _point_the_fence(monkeypatch, root=wt, live=live, worktree=True)
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: [("autonomy/9999-fence.md", doc)])

    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"autonomy/9999-fence.md::repo:{_TRACKED_TREE}/keep.py"}, (
        f"a file the round deleted was excused as out of scope: {sorted(drift)}")
    assert _OUT_OF_SCOPE == {}


def test_the_fence_is_closed_outside_a_worktree(tmp_path, monkeypatch):
    """Same trees, `IS_WORKTREE` off: the exemption is for a round whose base
    a rebase will move, never for a checkout that is simply behind. In `~/lloyd`
    the tree is what the vault was written against and every row counts."""
    live, wt = _live_tree_and_an_older_worktree(tmp_path)
    doc = _doc_naming(tmp_path, _LANDED_AFTER_BASE)
    _point_the_fence(monkeypatch, root=wt, live=live, worktree=False)
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: [("autonomy/9999-fence.md", doc)])

    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"autonomy/9999-fence.md::repo:{_LANDED_AFTER_BASE}"}
    assert _OUT_OF_SCOPE == {}


def test_a_fence_that_cannot_be_asked_exempts_nothing(tmp_path, monkeypatch):
    """Fail-closed: a live tree that shares no history with the worktree (or
    is not a repo at all) answers no merge-base, and the scan must count every
    row rather than widen because its oracle went quiet."""
    live, wt = _live_tree_and_an_older_worktree(tmp_path)
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    doc = _doc_naming(tmp_path, _LANDED_AFTER_BASE)
    _point_the_fence(monkeypatch, root=wt, live=stranger, worktree=True)
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: [("autonomy/9999-fence.md", doc)])

    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"autonomy/9999-fence.md::repo:{_LANDED_AFTER_BASE}"}
    assert _OUT_OF_SCOPE == {}


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


def test_a_new_untracked_naming_file_gets_its_own_diagnosis(tmp_path, monkeypatch):
    """#1411's second defect. The skew hint told a reader to `git show HEAD:<rel>`
    and, on 0, to commit or revert the vault file. For a doc in NO commit — a
    brand-new skill, `desktop-computer-use` all of 2026-09-23 — that grep is 0
    by construction and both remedies are wrong. The report has to say the file
    is untracked, must not offer the commit-or-revert line, and must still not
    call the row a missing file."""
    repo = _git_repo_with_committed_task(tmp_path)
    doc = repo / "autonomy" / "9998-new.md"
    doc.write_text(_task_body(names_script=True), encoding="utf-8")   # never added

    label = "autonomy/9998-new.md"
    monkeypatch.setattr(sys.modules[__name__], "_doc_files", lambda: [(label, doc)])
    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"{label}::repo:{_SKEW_SCRIPT}"}, sorted(drift)

    msg = _drift_report(drift)
    assert "UNTRACKED" in msg and label in msg, f"the text never says the file is new: {msg}"
    assert "committing or reverting" not in msg, (
        f"a file no commit holds was offered the tracked-file remedy: {msg}")
    assert _SKEW_SCRIPT in msg, f"the text never names the cited path to fix: {msg}"
    assert "not in this checkout" not in msg, (
        f"a working-tree skew was still reported as a missing file: {msg}")


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



# ── #1531 — a clipped path quotation is not a path being named ───────────────

def test_a_clipped_path_quotation_is_not_read_as_a_phantom(tmp_path, monkeypatch):
    """The guard reads a quotation that ran out of room as a claim about a short
    file, and reddens three nodes on a real tree with no defect in it.

    `clip_skill_description` cuts a description to a character budget and marks the
    cut with `…` (`prompt_builder.py:992-996`); `scripts/skill_lint.py:822` writes
    that clipped string into a cell of `~/obsidian/autonomy/skill-lint-report.md`,
    which is an autonomy task file and therefore in this guard's own corpus. On
    2026-09-25 the report's `service-health-check` row clipped mid-path and left
    `…~/lloyd/agent-services/su…`; the guard extracted `agent-services/su`, asked
    the checkout for it, found nothing, and filed the report as drift. That broke
    `test_no_active_skill_or_task_names_a_path_absent_from_the_checkout` outright,
    and because the fixture tests assert their planted row is the *only* one, it
    broke the two probes that exist to prove this check can still fail.

    Four shapes, both sides of the rule. The clipped pair must produce no row: one
    cut after the `~/lloyd/` anchor, one cut inside a backticked token (the two
    sites where a marker can arrive, since the anchored classes stop at `…` and the
    backticked class carries it through). The unclipped pair pins the direction that
    must survive: a real path is still not a violation, and an absent file named in
    full is still exactly one.
    """
    from prompt_builder import clip_skill_description

    real = "agent-services/supervisor/supervisord.conf"
    absent = "eval/a_script_only_a_clipped_quotation_names_1531.py"
    clipped_anchor = clip_skill_description(f"run ~/lloyd/{real} to check the conf", 40)
    assert "…" in clipped_anchor and real not in clipped_anchor, clipped_anchor

    docs = [
        ("autonomy/9101-anchor-clipped.md", f"| `svc` | clipped | {clipped_anchor} |\n"),
        ("autonomy/9102-backticked-clipped.md", "see `agent-services/su…` for the conf\n"),
        ("autonomy/9103-anchor-real.md", f"run ~/lloyd/{real} to check the conf\n"),
        ("autonomy/9104-anchor-absent.md", f"run ~/lloyd/{absent} to do the thing\n"),
    ]
    files = []
    for label, text in docs:
        p = tmp_path / label.split("/")[-1]
        p.write_text(text, encoding="utf-8")
        files.append((label, p))
    real_docs = _doc_files()
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: real_docs + files)

    rows = {r for r in _unresolved() if r.split("::", 1)[0] in {l for l, _ in docs}}
    assert rows == {f"autonomy/9104-anchor-absent.md::repo:{absent}"}, (
        f"a clipped quotation was read as a phantom, or an absent file named in full "
        f"went unseen: {sorted(rows)}")
    assert _truncated("agent-services/su", "… |") and _truncated("app/x…"), (
        "_truncated stopped covering one of the two arrival shapes")
