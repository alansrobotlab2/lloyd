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
and `AGENT_MENTION_EXEMPT`, both of which the tests below do consult. The path
half also exempts by *rule* where it can, so that no entry has to be maintained:
an ignored path (`_tree_ignores`), a path the document creates itself
(`_creation_sites`, #2157, #2223 — shell write sites and prose run records alike),
a template, a clipped quotation.
"""

from __future__ import annotations

import ast
import json
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
    from app.prompt_builder import _is_quarantined_skill

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

#: Where a document says it WRITES a path: a shell redirection, an output flag, or
#: an explicit `mkdir`. Read by `_declared_outputs`, which is what tells a path a
#: job creates apart from a path a job must open (#2157). `<` and a backtick are
#: out of the captured class, so a placeholder (`--report <path>`) yields nothing
#: usable rather than a shape that reads like a claim.
_OUTPUT_TARGET = re.compile(r"(?:>>?|--report|--output|--out|-o)\s+([^\s;|&)<`]+)")
_MKDIR_TARGET = re.compile(r"\bmkdir\s+(?:-{1,2}\w+\s+)*([^\s;|&)<`]+)")

#: Where a document says something is WRITTEN to a path in an English sentence
#: rather than at a shell write site (#2223). Read by `_prose_creation_sites`.
#: Creation verbs only, in any tense, and the path must be the token IMMEDIATELY
#: after the verb (one optional preposition, one optional opening quote), so
#: "read `~/lloyd/x`", "open `~/lloyd/x`" and "run `~/lloyd/x`" keep their rows.
#: `>`, `<`, `$` and `(` are outside the captured class, which is what makes a
#: placeholder (`<date>`) and a shell substitution (`$(date -u +%F)`) yield no
#: usable token instead of a shape that reads like a claim.
_PROSE_CREATION = re.compile(
    r"\b(?:wrote|written|writ(?:e|es|ing)|creat(?:e|es|ed|ing)|generat(?:e|es|ed|ing)"
    r"|sav(?:e|es|ed|ing)|emitt?(?:s|ed|ing)?|produc(?:e|es|ed|ing)"
    r"|append(?:s|ed|ing)?)\b"
    r"(?:\s+(?:to|into|onto|at|in|as|out))?"
    r"\s+[`'\"]?\s*([^\s;|&,)`'\"<>]+)", re.IGNORECASE)

#: Where a document quotes a raised exception whose payload IS the path
#: (`FileNotFoundError: /home/…/x`, `FileNotFoundError: [Errno 2] No such file or
#: directory: '/x'`) — #2223. Read by `_absence_record_sites`. Case-sensitive on
#: purpose: the exception's CamelCase is the whole signal that the text is a
#: runtime message and not the document's own citation. Nothing may sit between the
#: colon and the path except an errno bracket and quotes, so a message *about*
#: something else that happens to mention a path (`ValueError: bad key in
#: ~/lloyd/config.yaml`) captures nothing — only a message whose subject is the
#: path itself. The captured class is `_PROSE_CREATION`'s, so a placeholder or a
#: clipped quote yields no usable token rather than a shape that reads like a claim.
_ABSENCE_RECORD = re.compile(
    r"\b[A-Z][A-Za-z0-9_]{2,}(?:Error|Exception)\b"
    r":\s*"
    r"(?:\[[^\]\n]{1,40}\][^\n:]{0,40}:\s*)?"
    r"[`'\"]?\s*"
    r"([^\s;|&,)`'\"<>]+)")

#: Where a generated document states its own subject rather than citing a
#: location: the machine block
#: `scripts/maintenance/referential_integrity.py:327` appends to the
#: referential-integrity report — `<!-- ri:dangling ["<citing> -> <target>", ...] -->`,
#: the same block that script's own `_STATE_RE` (:236) reads back to diff one night's
#: report against the next. Read by `_dangling_subjects`. The report's entire content
#: is a list of cites that do not resolve, so a target it enumerates is a path the
#: document is ABOUT, not one it cites. Same polarity `_absence_record_sites` keys on
#: (who is talking), from the other side: there the speaker is a raised exception,
#: here it is a sweep that has just proved the path is gone. #2354: the run of that
#: job at 2026-10-07T12:00:54Z recorded `scripts/util/owed.py` and
#: `scripts/autonomy_status.py` as newly dangling, and the four path nodes went red on
#: the accuracy of that report, not on any drift.
_RI_DANGLING = re.compile(r"<!--\s*ri:dangling\s*(\[.*?\])\s*-->", re.S)

#: One line of that report's own enumeration: "- `<citing>:<line>` → `<target>` —
#: <how it was checked>", emitted for both the Dangling and the Exempt sections
#: (`referential_integrity.py:321` and `:325`). Read by `_dangling_subjects`, and only
#: in a document that also carries the block above — the block is what proves the
#: bullet list is a generated sweep and not a hand-written doc full of arrows. The
#: target is the right-hand side ONLY: the citing half is the other document's
#: problem, asked about by the sweep itself, and a bullet never cites a path it is
#: reporting on.
_LEDGER_BULLET = re.compile(r"^\s*-\s*`[^`\n]*`\s*(?:→|->)\s*`([^`\n]+)`", re.M)

#: The marker left where something was cut short. `clip_skill_description` says so
#: of itself — "cut to at most `max_chars` characters, the cut marked with `…`"
#: (`app/prompt_builder.py:992-996`) — and `scripts/skill_lint.py:822` puts that clipped
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
#:
#: The three docs that named `scripts/vault/okf_taxonomy.CANONICAL_TYPES` came
#: out of this set on 2026-09-30 (#1899). They were never drift: `okf_taxonomy.py`
#: is in the checkout and binds `CANONICAL_TYPES`, which is what
#: `_dotted_module_ref` now proves before it lets a dotted reference go. A dotted
#: reference whose module or member really is absent still arrives here as a red
#: row, so nothing is exempted by writing its shape off.
#:
#: Two `system-health-check` rows left on 2026-10-08 (#2422), together with the
#: 9-line keep-comment that sat between them. That comment had done useful work: it
#: recorded why `tests/test_system_health_check_frontend_endpoint.py` left this set on
#: 2026-10-03 (#2129) — the path is in the checkout now, and an allowance for a file
#: that exists is how a ledger starts lying about what is owed — and then declared
#: that the two rows it sat between stayed, "because those paths are still absent and
#: still cited, at `check-components.md:57` and `:61`". #2422 deleted those two lines:
#: the first asserted a re-check of the supervisor table by a test file this repo has
#: never contained, and the second named a file that does not exist either. With the
#: citations gone each row was excusing nothing, which is the exact state that
#: comment's own closing sentence said a row must not outlive. The reason this
#: deletion is safe to make in a set nothing counts is that the same retirement is
#: pinned from the enforced side. In the ledger's own test file,
#: `tests/test_fact_identity_one_action_one_fact.py`, the node
#: `test_the_absent_script_ledger_holds_exactly_the_cited_debts`
#: asserts the exact `KNOWN_ABSENT_SCRIPTS` mapping, so the retired entry cannot
#: come back without a red node somewhere.
PATH_KNOWN_UNFIXED: set[str] = {
    "skills/ai-engineer-monitor/SKILL.md::vault:autonomy/75-ai-engineer-youtube-monitor.md",
    "skills/documentation-digester/SKILL.md::repo:agent-services/llm/llama.cpp",
    "skills/documentation-digester/SKILL.md::repo:agent-services/llm/llama.cpp/build/bin/llama-server",
    "skills/entity-resolution-sweep/SKILL.md::repo:agent_mcp/memory.py",
    "skills/file-path-resolution/SKILL.md::repo:inner-voice/system_prompt.md",
    "skills/file-path-resolution/SKILL.md::repo:lloyd/inner-voice/system-prompt.md",
    "skills/file-path-resolution/SKILL.md::vault:lloyd/inner-voice/system-prompt.md",
    "skills/file-read-resilience/SKILL.md::vault:agents/idler/gateway.py",
    "skills/historical-knowledge-refresh/SKILL.md::repo:scripts/memory/extract-session-log.py",
    "skills/iv-plan-review/SKILL.md::repo:app/middleware/session_auth.py",
    "skills/memory-path-scoping/SKILL.md::repo:scripts/memory/next-gen-memory/context_bundle.py",
    "skills/plan-mode-authoring/SKILL.md::repo:app/middleware/session_auth.py",
    "skills/plan-mode-authoring/SKILL.md::repo:web/src/components/TagChip.tsx",
    "skills/poisoned-worker-troubleshoot/SKILL.md::vault:logs/autonomy_runs/run_",
    "skills/powerpoint/SKILL.md::repo:scripts/office/soffice.py",
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


#: What may follow the last dot of a `app/module.MEMBER` reference: an
#: identifier. Anything else (`.cpp`, `.tsx`, `.wav`) is a file extension and
#: stays a file claim, which is what keeps `references/ed/ed_001.wav` and
#: `web/src/components/TagChip.tsx` red under this rule.
_MEMBER_TAIL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def _bound_names(module: Path) -> frozenset[str]:
    """The names `module` binds at its own top level, read from its AST.

    Definitions, classes, assignments and imports — the module's own namespace,
    which is what a reader following `app/module.MEMBER` finds. Deliberately not
    a grep: a comment or a docstring that mentions `MEMBER` does not make the
    reference resolvable. A name bound only inside a `try:` or an `if` is not
    top-level and stays red, which is the pre-existing verdict for it, not a new
    one. An unreadable file or one that will not parse answers nothing, and
    nothing exempts — the fail-closed rule `_tree_ignores` follows.
    """
    try:
        tree = ast.parse(module.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError, RecursionError):
        return frozenset()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                out.update(n.id for n in ast.walk(tgt) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add(alias.asname or alias.name.split(".")[0])
    return frozenset(out)


def _dotted_module_ref(rel: str, candidates: list[Path]) -> bool:
    """True when `rel` is a Python dotted reference the checkout really resolves.

    `candidates` is the same list the file-existence probe just refused — one
    place the named file would have been, per root the scan knows.

    A doc that names ``app/run_acceptance.grade_run`` points at a member of
    `app/run_acceptance.py`, not at a file called `run_acceptance.grade_run` —
    a name no checkout can contain, because Python modules are files with ONE
    dot. `BACKTICKED` takes the whole token and the existence probe then looks
    for that file, so a reference that is correct Python was reported as drift:
    `autonomy/76-queue-health-check.md` reddened three nodes at base on
    2026-09-30 (item #1899), and three `PATH_KNOWN_UNFIXED` entries for
    `scripts/vault/okf_taxonomy.CANONICAL_TYPES` are the same shape, grandfathered
    on 2026-09-18 rather than fixed.

    This is a resolution, not an exemption, and that is the difference from the
    arrow form #1454 refused: nothing here matches a *syntactic* shape and lets
    it through. The rule fires only when the tree proves BOTH halves — the module
    file exists AND it binds that name at its own top level — so a phantom member
    of a real module stays red. That is the #1240 defect one level deeper than
    the missing script: instructions naming something no run can call. Exemptions
    this file already accepts widen on a property of the tree (`_tree_ignores`)
    or on the text not being a claim at all (`_truncated`); this one narrows on
    the module's own contents.

    Last dot only, so `app/pkg/mod.ATTR` and `app/pkg.ATTR` (through a package
    `__init__.py`) resolve and a two-member tail like `app/mod.Class.method`
    keeps the old verdict rather than guessing. `repo:` rows only — a Python
    module lives in the checkout, and a `vault:` row's tree is the vault.
    """
    head, _, member = rel.rpartition(".")
    if not head or "/" not in head or not _MEMBER_TAIL.fullmatch(member):
        return False
    stem = head[head.rindex("/") + 1:]
    for named in candidates:
        # `named` is where the reference said the FILE would be, one candidate per
        # root the scan probes (repo root, then the skill's own folder), so the
        # module sits beside it under its own name — `app/pkg/mod.ATTR` asks
        # `app/pkg/mod.py`, and a skill-local `scripts/pkg.ATTR` asks inside the
        # skill, exactly as the file-existence probe two lines above does.
        module = named.parent / f"{stem}.py"
        if not module.is_file():
            module = named.parent / stem / "__init__.py"
        if module.is_file() and member in _bound_names(module):
            return True
    return False


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


def _declared_outputs(body: str) -> set[tuple[str, str]]:
    """(tree, rel) for each path the document says it CREATES in its own text.

    #85's defect was a run that could not open a file its instructions named, so
    this fence asks one question of a named path: is it on disk. That question is
    only well-posed of a path the doc *reads*. A job that writes a report has to
    name the path it writes before the report exists — that is what a `--report`
    argument and a `mkdir -p` are for — and on the night such a job is filed its
    output is by definition absent, which is the red node that went out over
    `referential-integrity-ledger` (#2157): two committed docs, three nights of
    rounds refused by their own not-yet-run outputs.

    So an absent path is not drift when the same document puts it at a write
    site. Deliberately narrow:

      * the creation site must be in the **same document** — a `mkdir` anywhere
        else in the corpus exempts nothing, so this cannot become a corpus-wide
        list of paths nobody has to maintain;
      * the path must be checkout- or vault-rooted (`~/lloyd/…`, `~/obsidian/…`),
        which is the same root claim `_named_paths` requires; `$HOME/…`, `./out`
        and a bare word normalize to nothing and stay drift;
      * it is an **exact** path, not a prefix: `mkdir -p ~/obsidian/autonomy`
        would otherwise exempt every report anyone files under `autonomy/`.

    Fail-closed on anything odd: an unparseable token, a placeholder (`<path>`,
    `$(date)`) and a clipped quote all yield no entry, so the worst a malformed
    creation statement can do is leave the row it would have left.
    """
    out: set[tuple[str, str]] = set()
    for pat in (_OUTPUT_TARGET, _MKDIR_TARGET):
        for tok in pat.findall(body):
            for rx, tree in ((_LLOYD_PATH, "repo"), (_OBSIDIAN_PATH, "vault")):
                m = rx.fullmatch(tok)
                if m is not None and not _truncated(m.group(1)):
                    out.add((tree, m.group(1)))
    return out


def _prose_creation_sites(body: str) -> set[tuple[str, str]]:
    """(tree, rel) for each path a document names as the thing its subject WRITES,
    stated in prose rather than at a flag, a redirection or a `mkdir`.

    #2157 taught the fence that a path the doc *creates* is an output and not a
    citation, but only read it off shell syntax, which is the shape a command puts
    an output in. A task file recording what a finished run did states the same
    fact in English. `autonomy/96-djev-name-prior-probe.md` has "It wrote
    `/home/alansrobotlab/lloyd/eval/djev/name_prior_2026-10-04.json` and left
    `name_prior_2026-09-24.json` untouched": a history sentence about the report
    the probe produced, which is untracked in the code tree because the task's own
    Never list bars it from `git add`. A worktree never holds an untracked file, so
    that true sentence was a dead-path row, and because the report is dated the row
    arrived again every week — the three nodes that assert their own planted
    violation is the SOLE drift went red with it (#2223).

    The same three scopes `_declared_outputs` keeps, for the same reasons: the
    statement must be in the **same document**; the path must be checkout- or
    vault-rooted, so `$HOME/…`, `./out` and a bare relative word normalize to
    nothing and stay drift; and it must be an **exact** path, so a sentence about
    writing a directory exempts no file inside it. Fail-closed on anything odd: a
    placeholder (`<date>`), a shell substitution and a clipped quote each yield no
    entry, so the worst a malformed sentence can do is leave its own row.

    What this is NOT is a tense detector: it reads one statement type, and a
    document recording that a path was REMOVED or swept is outside it. #2026 ruled
    on that shape — an incident note quoting two strays the sweep had deleted, each
    sentence true and each path dead — and the remedy was to reword the prose, which
    is still what `scripts/util/skill_path_findings.py` advises. A deletion record
    names the path as its subject's *content*; a creation record names it as the
    subject's *product*, and only the second one has to be written before it can be
    read. The polarity question that is NOT settled by tense — a passage whose
    speaker is a raised exception, saying in so many words that the path could not
    be opened — is `_absence_record_sites`, which keys on who is talking rather
    than on when.
    """
    out: set[tuple[str, str]] = set()
    for m in _PROSE_CREATION.finditer(body):
        for rx, tree in ((_LLOYD_PATH, "repo"), (_OBSIDIAN_PATH, "vault")):
            full = rx.fullmatch(m.group(1))
            if full is not None and not _truncated(full.group(1)):
                out.add((tree, full.group(1)))
    return out


def _creation_sites(body: str) -> set[tuple[str, str]]:
    """Every path the document says it puts somewhere: shell write sites plus
    prose creation records (#2157, #2223).

    One definition of the creation half, joined to the absence-record half by
    `_recorded_sites`, which is the name
    `scripts/util/skill_path_findings.py` loads to keep its report leg from printing
    a finding for a path the node exempts: two copies of "does this sentence cite a
    living path, or record one?" is how the writer and the node start disagreeing
    about the same document.
    """
    return _declared_outputs(body) | _prose_creation_sites(body)


def _absence_record_sites(body: str) -> set[tuple[str, str]]:
    """(tree, rel) for each path that appears ONLY as the payload of a quoted
    exception — text whose speaker is a runtime message, not the document.

    `autonomy/76-queue-health-check.md`'s Activity Log carries, from
    `run_76_20261005_130005`: "…shows the errored call: `FileNotFoundError:
    /home/alansrobotlab/obsidian/memory/scratchpads/2f363115-b9bf-4cfa-8c4f". The
    node read that as drift: a doc naming a path that is not on disk. But the
    sentence is the fence's own verdict, already published — the file could not be
    opened — and a check that goes red because a document faithfully reported an
    absence is red about the report, not about the path. This is the polarity half
    of #2223, the one tense cannot carry: "It wrote `x`" and "`FileNotFoundError:
    x`" are both past tense and opposite in direction, so the rule keys on who is
    talking.

    It also cannot be fixed at the source the way #2026's incident note could. That
    line is one bullet of an Activity Log the autonomy harness appends after every
    run of task 76, and the writer clips the bullet mid-path with no `…` to mark it
    (the row ends `-c4f` where the file it quotes was `…-c4f.md`), so no rewording
    survives the next run. A hand-kept entry would be worse: `PATH_KNOWN_UNFIXED`
    cannot hold a UUID per incident (#2207's own lesson for this corpus).

    Narrowed to the payload position, and fail-closed like its siblings: the path
    must be checkout- or vault-rooted and exact, so a relative word, `$HOME/…` or a
    placeholder exempts nothing; the exception must name the path as its whole
    payload (an errno bracket and quotes are the only thing allowed between the
    colon and it), so a message that merely mentions a path keeps its row; and the
    exemption is per-path in the same document, so a page that quotes one error and
    then tells a run to open the same missing file is caught by that second
    sentence.
    """
    out: set[tuple[str, str]] = set()
    for m in _ABSENCE_RECORD.finditer(body):
        for rx, tree in ((_LLOYD_PATH, "repo"), (_OBSIDIAN_PATH, "vault")):
            full = rx.fullmatch(m.group(1))
            if full is not None and not _truncated(full.group(1)):
                out.add((tree, full.group(1)))
    return out


def _dangling_targets(body: str) -> set[tuple[str, str]]:
    """(tree, rel) for each path a referential-integrity report is ABOUT — its own
    enumeration, read from its `ri:dangling` machine block and from the bullets beside
    it — and only once that block parses as a JSON list (#2354).

    A document whose block is missing or unreadable gets nothing at all, so a
    hand-written note full of arrows changes nothing and a half-written report cannot
    exempt a path by being malformed: the block, emitted by
    `scripts/maintenance/referential_integrity.py:327` and read back by that script's
    own `_STATE_RE` (:236) to diff one night against the next, is what proves the bullet
    list is a generated sweep rather than prose a reader follows. Inside such a document
    BOTH surfaces are its subject, because they are the same enumeration — the machine
    block carries only the day's dangling keys while the bullets also carry the Exempt
    section (`:325`), and a path the sweep filed as exempt is a path it proved absent
    and then decided was nobody's cite. Reading only the block would leave the other
    half of that document's own list to go red on the next sweep, which is the failure
    this rule exists to end. Same polarity as `_absence_record_sites` (who is talking),
    from the other side: there the speaker is a raised exception naming a missing file,
    here a sweep that has just proved the path is gone from every root it could sit in.

    Narrowed to fail closed like its siblings. Only the target half of a
    `"<citing> -> <target>"` entry counts, so the document a row is ABOUT is never
    exempted by it, and a bullet's left-hand `` `citing:line` `` is never a target. A
    target is classified the way `_named_paths` classifies a backticked token, so it
    lands in the ONE tree it is spelled in: a report listing `lloyd/x.py` does not also
    excuse a skill's claim about `obsidian/x.py`, and a bare `scripts/x.py` cannot
    claim the vault half. A clipped target, a non-string entry, a block that is not a
    JSON list and a target the scanner could not have produced each yield nothing, so
    the worst a malformed report can do is leave its own rows red.

    What this does NOT do is silence the document. Its PROSE stays citations: the line
    naming the instrument that wrote it
    (`scripts/maintenance/referential_integrity.py`) and any path its sentences name are
    still rows if those paths go, which
    `test_what_a_dangling_ledger_lists_makes_no_row_and_what_its_prose_names_does` pins
    on a fixture and `test_the_live_dangling_report_loses_its_own_rows_and_nothing_else`
    pins on the real report. And it is per-document: `_recorded_sites` is called one body
    at a time, so a skill may still drift a path the ledger happens to list, which
    `test_a_path_the_ledger_lists_still_reads_as_drift_in_a_skill` pins.
    """
    m = _RI_DANGLING.search(body)
    if m is None:
        return set()
    try:
        entries = json.loads(m.group(1))
    except ValueError:
        return set()
    if not isinstance(entries, list):
        return set()
    out: set[tuple[str, str]] = set()
    for raw in _LEDGER_BULLET.findall(body):
        pair = _classify_target(raw)
        if pair is not None:
            out.add(pair)
    for entry in entries:
        if not isinstance(entry, str) or " -> " not in entry:
            continue
        pair = _classify_target(entry.split(" -> ", 1)[1])
        if pair is not None:
            out.add(pair)
    return out


def _classify_target(raw: str) -> tuple[str, str] | None:
    """`(tree, rel)` for one path string a report names, resolved the way
    `_named_paths` resolves a backticked token — rooted in the checkout, rooted in the
    vault, then a bare `lloyd/…`/`skills/…` string as checkout-only. `None` for anything
    the scanner could not have produced as a row, so it exempts nothing: a root the
    test does not model (`~/lloyd-data/…`, `$HOME/…`), a bare name with no tree-bearing
    prefix, a clipped path."""
    target = raw.strip().strip("`")
    if not target or _truncated(target):
        return None
    for rx, tree in ((_LLOYD_PATH, "repo"), (_OBSIDIAN_PATH, "vault")):
        full = rx.fullmatch(target)
        if full is not None:
            return (tree, full.group(1))
    if target.startswith(CHECKOUT_PREFIXES) or target.startswith(SKILL_LOCAL_PREFIXES):
        return ("repo", target)
    return None


def _recorded_sites(body: str) -> set[tuple[str, str]]:
    """Paths the document mentions without citing them as living somewhere: what
    its subject writes (`_creation_sites`, #2157 + #2223), what a quoted exception
    says is missing (#2223), and what a referential-integrity report enumerates as its
    own subject (`_dangling_targets`, #2354).

    The one name `_absent_refs` consults, and the one name
    `scripts/util/skill_path_findings.py` subtracts from its bench leg, so the
    writer-side CHECK and the unmarked node cannot diverge on a sentence. Adding a
    mention type means adding it here, once.
    """
    return (_creation_sites(body) | _absence_record_sites(body)
            | _dangling_targets(body))


def _absent_refs(label: str, body: str, skill_dir: Path | None) -> set[str]:
    """`<label>::<tree>:<path>` for each path `body` names that is not on disk.

    One document's worth of `_unresolved_rows`, before the two tree-level
    exemptions (ignore rules, landed-after-base) are applied. Split out so the
    writer side can ask the same question of a body that is not on disk yet:
    `scripts/util/skill_path_findings.py` runs this over the STAGED text of a
    skill inside `vault-commit.sh`, which is where a nightly job that re-adds a
    removed path is told so before its commit turns this file red on main
    (#1969). One rule, two callers — a second copy of it there would drift.
    """
    out: set[str] = set()
    recorded: set[tuple[str, str]] | None = None
    for tree, rel in _named_paths(body, skill_dir):
        if _is_template(rel) or "/" not in rel or rel.endswith(RUNTIME_SUFFIXES):
            continue
        if tree == "vault":
            roots = [VAULT / rel]
        else:
            roots = [ROOT / rel]
            if skill_dir is not None:
                roots.append(skill_dir / rel)
        if any(r.exists() for r in roots):
            continue
        # Only an absent path is asked whether the doc records it, so the green
        # path stays free of this scan — the rule `_DIRTY_CACHE` follows. A
        # creation site counts whether it is written as a flag, a redirection, a
        # `mkdir` or an English sentence, and an absence record counts a path a
        # quoted exception says could not be opened (#2157, #2223).
        if recorded is None:
            recorded = _recorded_sites(body)
        if (tree, rel) in recorded:
            continue
        if tree == "repo" and _dotted_module_ref(rel, roots):
            continue
        out.add(f"{label}::{tree}:{rel}")
    return out


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

    A `repo:` path that is absent AS A FILE is not yet drift: `_dotted_module_ref`
    asks the checkout whether the reference instead resolves as a Python dotted
    name, and only an answer of "no module, or no member" makes it a row (#1899).
    A `vault:` row is never asked, because a Python module lives in the checkout.
    """
    _OUT_OF_SCOPE.clear()
    bad: dict[str, Path] = {}
    for label, path in _doc_files():
        body = path.read_text(encoding="utf-8", errors="replace")
        skill_dir = path.parent if label.startswith("skills/") else None
        for row in sorted(_absent_refs(label, body, skill_dir)):
            bad[row] = path
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


#: A member `keep.py` binds nowhere, so the row for it must survive the dotted rule.
_PHANTOM_MEMBER = "not_bound_anywhere_1899"


def _fixture_checkout_with_python_modules(tmp_path: Path) -> Path:
    """A stand-in checkout holding one module, one package, and nothing else.

    Real files in a real tree, not a mocked namespace: the dotted reference is
    resolved by reading a module's AST off disk, so the control has to cross that
    boundary — the same reason `_fixture_checkout_with_its_own_ignore_rules` runs
    a real `git init` rather than stubbing the ignore answer.
    """
    repo = tmp_path / "checkout"
    (repo / _TRACKED_TREE).mkdir(parents=True)
    (repo / _TRACKED_TREE / "keep.py").write_text(
        "keep_me = 1\n\n\ndef keep_fn():\n    return 2\n", encoding="utf-8")
    (repo / _TRACKED_TREE / "pkg_1899").mkdir()
    (repo / _TRACKED_TREE / "pkg_1899" / "__init__.py").write_text(
        "pkg_member = 3\n", encoding="utf-8")
    (repo / ".gitignore").write_text("/ignored_by_the_tree/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_a_dotted_reference_resolves_only_when_its_module_and_its_member_do(
        tmp_path, monkeypatch):
    """#1899, all four verdicts in ONE scan of one fixture.

    A doc naming `tracked_source/keep.keep_me` names a member of a module the
    checkout has — `BACKTICKED` handed the whole token to a probe that then looked
    for a FILE called `keep.keep_me`, which no checkout can contain. The two
    resolved rows and the two unresolvable ones are planted in the same document
    and read by the same call, and the assertion is `==`, so the rule cannot widen
    without failing here: a scanner that excused dotted references by *shape*
    would drop the phantom-member row, and #1454 is the precedent for refusing
    that (a doc that merely looks self-correcting carries a phantom into every run
    that reads it). A phantom member of a real module is that defect one level
    deeper — instructions naming something no run can call.
    """
    repo = _fixture_checkout_with_python_modules(tmp_path)
    doc = tmp_path / "9999-dotted.md"
    doc.write_text(
        "---\nname: dotted\ntype: autonomy\n---\n# Dotted Task\n\n"
        f"Step 1 names a member the module binds: `~/lloyd/{_TRACKED_TREE}/keep.keep_me`.\n\n"
        "Step 2 names a member of the same module that binds no such name: "
        f"`~/lloyd/{_TRACKED_TREE}/keep.{_PHANTOM_MEMBER}`.\n\n"
        "Step 3 names a member of a module the tree has never held: "
        f"`~/lloyd/{_TRACKED_TREE}/no_such_module_1899.keep_me`.\n\n"
        f"Step 4 names a member through a package: `~/lloyd/{_TRACKED_TREE}/pkg_1899.pkg_member`.\n",
        encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", repo)
    monkeypatch.setattr(sys.modules[__name__], "_doc_files",
                        lambda: [("autonomy/9999-dotted.md", doc)])

    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == {f"autonomy/9999-dotted.md::repo:{_TRACKED_TREE}/keep.{_PHANTOM_MEMBER}",
                     f"autonomy/9999-dotted.md::repo:{_TRACKED_TREE}/no_such_module_1899.keep_me"}, (
        f"the dotted rule resolved or refused the wrong half of the four refs: {sorted(drift)}")
    assert _OUT_OF_SCOPE == {}, "the dotted refs were set aside rather than resolved"


def test_the_dotted_corpus_references_resolve_by_the_rule_and_not_by_a_ledger_entry():
    """#1899 against the real trees, because a fixture green proves only the rule
    exists — not that it fires on the four references that made this necessary.

    Three of them (`scripts/vault/okf_taxonomy.CANONICAL_TYPES`, in
    `deep-research`, `medium-research` and `quick-research`) sat in
    `PATH_KNOWN_UNFIXED` as debt since 2026-09-18; the fourth,
    `app/run_acceptance.grade_run`, reddened three nodes at base on 2026-09-30.
    All four leave this file's view because the checkout answers for them, so the
    ledger must stay empty of them — an entry there would be the #1317 failure
    shape again, a property re-broken by hand-writing its exceptions. And the
    scanner must still SEE the reference: a green that came from the pattern
    missing it would look identical here and be worth nothing, which is what
    `test_the_path_check_anchored_to_a_path_that_really_exists` is for one level up.
    """
    ref = "scripts/vault/okf_taxonomy.CANONICAL_TYPES"
    for skill in ("deep-research", "medium-research", "quick-research"):
        skill_dir = VAULT / "skills" / skill
        refs = {rel for _tree, rel in _named_paths(
            (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace"),
            skill_dir)}
        assert ref in refs, (
            f"{skill} no longer names the dotted reference this rule exists for, so "
            f"the green below proves nothing; got {sorted(refs)}")

    module = ROOT / "scripts/vault/okf_taxonomy.py"
    assert module.is_file() and "CANONICAL_TYPES" in _bound_names(module), (
        f"the module the three skills name ({module}) stopped binding the member "
        "they name — those rows are real drift again, and the right fix is the docs")

    rows = {r for r in _unresolved() if r.endswith(f"::repo:{ref}")}
    assert not rows, (
        f"the rule stopped resolving live docs, and three skills are red at base "
        f"again: {sorted(rows)}")
    assert not [e for e in PATH_KNOWN_UNFIXED if "okf_taxonomy" in e], (
        "the dotted references were exempted by enumeration instead of resolved: "
        "a row the rule already resolves is sitting in PATH_KNOWN_UNFIXED")
    assert "autonomy/76-queue-health-check.md::repo:app/run_acceptance.grade_run" \
        not in _unresolved(), "#1899's own row is back"


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


#: What a task file must contain to count as an instruction for this eval. Scanned
#: for the instrument rather than for a task id, because the id is retired.
EVAL_SCRIPT_NAME = "secondary_routing_eval.py"
#: The slug the skill loads under, and the door a scheduled task routes through.
EVAL_SKILL_NAME = "secondary-routing-eval"


def _routing_docs() -> dict[str, str]:
    """Every document that tells this eval's run what to do, keyed by label.

    The skill and an autonomy task are the same run's instructions read in either
    order, so a sentence corrected in one and not the other is a sentence the run
    will follow differently depending on which it opened — which is why this returns
    a dict and every node below loops over it rather than naming two paths.

    Task #85 was retired by Alan's ruling of 2026-09-26 (backlog **#1577**) and its
    file deleted, so today the dict holds the skill alone. The fleet is scanned
    rather than the path hardcoded because re-adding a task is the cheap move the
    ruling names, and a helper that quietly stopped matching would drop the pair back
    to one document with the suite green: a re-armed task is caught by this same loop.
    """
    docs = {
        "skill": (VAULT / "skills" / EVAL_SKILL_NAME / "SKILL.md").read_text(
            encoding="utf-8", errors="replace"),
    }
    docs.update(_tasks_naming_eval(VAULT / "autonomy"))
    return docs


def _tasks_naming_eval(autonomy_dir: Path) -> dict[str, str]:
    """Task files under `autonomy_dir` whose body names the eval instrument.

    Top-level `*.md` on purpose: that is the same non-recursive directory scan
    `app.autonomy.load_tasks` performs, so `_archived/` cannot smuggle a retired task
    back in, and it reads every status rather than only armed ones — stricter than the
    scheduler, which is the direction a witness should err in.
    """
    return {f"task ({path.name})": path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(autonomy_dir.glob("*.md"))
            if EVAL_SCRIPT_NAME in path.read_text(encoding="utf-8", errors="replace")}


def test_the_routing_doc_scan_finds_a_task_that_names_the_instrument(tmp_path):
    """The scan's positive control, which the deleted `85-*.md` glob did not need.

    `next(iter(glob("85-*.md")))` raised `StopIteration` the moment its document went
    away, so a vanished subject could not pass unnoticed. A scan that matches nothing
    looks exactly like a clean fleet, so this builds the task the scan is supposed to
    survive: if the fragment, the directory or the reader breaks, the pair silently
    returns to one document and every assertion below quietly applies to the skill
    alone. A decoy file is scanned alongside it to prove the fragment is doing the
    work and not "any file in the fleet".
    """
    (tmp_path / "97-rerouted.md").write_text(
        f"---\nskill_name: {EVAL_SKILL_NAME}\n---\n"
        f"runs eval/{EVAL_SCRIPT_NAME} nightly\n", encoding="utf-8")
    (tmp_path / "98-unrelated.md").write_text(
        "---\nskill_name: heartbeat\n---\nnothing to do with routing\n",
        encoding="utf-8")

    found = _tasks_naming_eval(tmp_path)

    assert list(found) == ["task (97-rerouted.md)"], (
        f"the scan returned {sorted(found)}: one document of two is what a broken "
        "scan reports, and the nodes reading _routing_docs() cannot tell that from a "
        "fleet that has no task")


def test_the_routing_skill_states_the_real_per_job_route():
    """Step 4 used to say a flip is made "in `config.yaml`" and that "a per-job
    route … does not exist yet". Both were false the moment
    `JOBS_ON_PRIMARY`/`_engine_for` landed: `config.yaml` has exactly two model
    slots, `primary` and `secondary`, and no per-job key, and following that
    sentence would have had a nightly set `secondary_enabled: false` — taking
    the engine down for all four jobs that were measured as keeps.

    Every document is held to it, not just the skill: a task file repeats the
    procedure for the run that reads the task row instead, and it named no
    route at all until item #1240, which left "how do I apply a flip"
    unanswered exactly where a flip is what the nightly is looking for. Task #85
    was retired (#1577), so the loop holds one document today and picks the pair
    back up if a task is ever re-added.
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


def test_the_nightly_instructions_name_one_location_for_the_trend():
    """Every instruction document for this eval, read through the one helper.

    They disagreed before: the skill said append to `eval/secondary-routing/trend.md`,
    the task said the same, and the four real runs wrote the vault instead — so
    whichever path this round picked, the documents had to move together or the next
    night drifted again. The node is named for the invariant rather than for the
    document count because #1577 deleted the task half: the count is whatever the
    fleet currently holds, and the one-location rule applies to each of them."""
    for name, doc in _routing_docs().items():
        assert "projects/lloyd/secondary-routing-eval/trend.md" in doc, (
            f"the {name} must point the trend row at the vault note that holds it")
        assert "eval/secondary-routing/trend.md" not in doc, (
            f"the {name} still names a repo trend file that nothing writes")



# ── #1531 — a clipped path quotation is not a path being named ───────────────

def test_a_clipped_path_quotation_is_not_read_as_a_phantom(tmp_path, monkeypatch):
    """The guard reads a quotation that ran out of room as a claim about a short
    file, and reddens three nodes on a real tree with no defect in it.

    `clip_skill_description` cuts a description to a character budget and marks the
    cut with `…` (`app/prompt_builder.py:992-996`); `scripts/skill_lint.py:822` writes
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
    from app.prompt_builder import clip_skill_description

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


# ---------------------------------------------------------------------------
# #2157: a path the document says it CREATES is an output, not a citation
# ---------------------------------------------------------------------------

# Three vault paths, none of which exists on this machine (`~/obsidian/reports`
# is not a directory), so every row below is a row about absence and not about
# whatever a previous run left behind.
_WRITTEN = "~/obsidian/reports/a_ledger_report_the_job_writes_2157.md"
_MKDIRS = "~/obsidian/reports/a_copy_dir_the_job_mkdirs_2157"
_READS = "~/obsidian/reports/an_input_the_job_only_reads_2157.md"
_WRITTEN_ROW = "skills/2157-writer/SKILL.md::vault:reports/a_ledger_report_the_job_writes_2157.md"
_MKDIRS_ROW = "skills/2157-writer/SKILL.md::vault:reports/a_copy_dir_the_job_mkdirs_2157"
_READS_ROW = "skills/2157-writer/SKILL.md::vault:reports/an_input_the_job_only_reads_2157.md"


def _writer_body(report: str = _WRITTEN, target: str = _MKDIRS,
                 reads: tuple[str, ...] = ()) -> str:
    """A job instruction in the shape `referential-integrity-ledger` actually has.

    One command whose script takes `--report <path>`, one `mkdir -p` of the
    directory the dated copy goes in, and any file the job is told to read. Both
    write sites are the ones that went out in the real skill, verbatim in shape:
    `#2157`'s rows came from exactly these two commands naming files the job had
    not yet had a night to produce.
    """
    lines = [
        "# Ledger job",
        "",
        "1. `cd ~/lloyd && python3 -m scripts.maintenance.referential_integrity"
        f" --report {report}; echo EXIT=$?`",
        f"2. `mkdir -p {target} && cp {report} {target}/$(date -u +%F).md`",
    ]
    lines += [f"{i}. Read {p} first." for i, p in enumerate(reads, start=3)]
    return "\n".join(lines) + "\n"


def test_a_path_the_document_creates_is_not_drift_and_one_it_reads_is():
    """Both directions of the exemption in one assertion, because the half that
    must keep biting is the read path.

    `_WRITTEN` is named at a `--report` site and `_MKDIRS` at a `mkdir -p`, so
    neither is drift; `_READS` is named only as something to open, which is #85's
    defect and still exactly one row. The whole body is one doc, so a rule that
    exempted too much would show up here as an empty set.
    """
    rows = _absent_refs("skills/2157-writer/SKILL.md",
                        _writer_body(reads=(_READS,)), None)
    assert rows == {_READS_ROW}, (
        f"a job's own outputs were read as drift, or a path it only reads went "
        f"unseen: {sorted(rows)}")


def test_the_creation_must_be_in_the_same_document_and_at_that_exact_path():
    """The two ways this exemption could quietly rot into a list nobody maintains.

    Naming the same path without creating it is drift again — `_writer_body` puts
    a `mkdir` and a `--report` in one document, and that document's declaration
    exempts nothing for any other document, so there is no corpus-wide set of
    outputs to keep current. And `mkdir -p ~/obsidian/reports` does not exempt a
    file under `reports/`: the exemption is an exact path, because a parent
    directory named once would otherwise cover every report anyone files there.
    """
    named_only = _absent_refs("autonomy/9157-reader.md",
                              f"The report lives at {_WRITTEN}\n", None)
    assert named_only == {
        "autonomy/9157-reader.md::vault:reports/a_ledger_report_the_job_writes_2157.md"}, (
        f"a document that only names a path stopped being asked about it: "
        f"{sorted(named_only)}")

    parent_mkdir = _absent_refs(
        "skills/2157-writer/SKILL.md",
        f"1. `mkdir -p ~/obsidian/reports`\n2. The report is {_WRITTEN}\n", None)
    assert parent_mkdir == {_WRITTEN_ROW}, (
        f"a mkdir of the parent directory exempted a file inside it: "
        f"{sorted(parent_mkdir)}")


@pytest.mark.parametrize("site", [
    "<path>",                                     # a placeholder, not a claim
    "$(date -u +%F).md",                          # a shell substitution
    "$HOME/obsidian/reports/a_ledger_report_the_job_writes_2157.md",
    "./reports/a_ledger_report_the_job_writes_2157.md",
])
def test_a_write_site_that_is_not_a_rooted_path_exempts_nothing(site):
    """Fail-closed: the write site counts only when it names a checkout- or
    vault-rooted path, which is the same root claim `_named_paths` requires.

    `$HOME/obsidian/…` is the interesting row: the *scanner* does read it as a
    vault path (its lookbehind does not exclude `$`), so a rule keyed on the
    string rather than on a resolved `(tree, rel)` would exempt it and lose a
    real drift row. `$HOME/…` and `./…` normalize to nothing and the row stands,
    which is why `_declared_outputs` runs its tokens through the two anchored
    matchers instead of comparing text.
    """
    body = f"1. `mkdir -p {site}`\n2. The report is {_WRITTEN}\n"
    rows = _absent_refs("skills/2157-writer/SKILL.md", body, None)
    assert rows == {_WRITTEN_ROW}, (
        f"a write site that is not a rooted path ({site!r}) exempted an absent "
        f"file: {sorted(rows)}")


def test_the_ledger_job_that_made_main_red_is_exempt_by_its_own_write_sites():
    """The corpus that made three nodes red at `4f015e9f`, checked in the tree.

    `skills/referential-integrity-ledger/SKILL.md` and
    `autonomy/94-referential-integrity-ledger.md` — both committed to the vault,
    both filed 2026-10-03 — name `autonomy/referential-integrity-latest.md` and
    `autonomy/referential-integrity/`, which the job writes with `--report` and
    creates with `mkdir -p` on its first night. The first assertion is the one
    that was red: no `referential-integrity` row survives in the live corpus. The
    second credits the exemption to those documents' own write sites rather than
    to a new entry in any ledger, and asks only files that are still in the
    corpus, so a later rewrite of the job cannot break it by deleting something.
    """
    drift = {r for r in _unresolved() if "referential-integrity" in r}
    assert drift == set(), (
        f"a job's not-yet-written outputs are being counted as drift again: "
        f"{sorted(drift)}")

    for name in ("skills/referential-integrity-ledger/SKILL.md",
                 "autonomy/94-referential-integrity-ledger.md"):
        doc = VAULT / name
        if not doc.exists():
            continue
        declared = _declared_outputs(doc.read_text(encoding="utf-8",
                                                    errors="replace"))
        assert ("vault", "autonomy/referential-integrity-latest.md") in declared, name
        assert ("vault", "autonomy/referential-integrity") in declared, name


# ---------------------------------------------------------------------------
# #2223: a prose record that the job WROTE its report is a creation site too
# ---------------------------------------------------------------------------

# Two checkout paths under the directory the name-prior probe really writes
# into, neither of which is in the tree, so every row below is a row about
# absence and not about whatever a previous run left behind.
_WROTE_2223 = "~/lloyd/eval/djev/a_created_2223.json"
_OPEN_2223 = "~/lloyd/eval/djev/a_read_2223.json"
_WROTE_2223_PATH = "eval/djev/a_created_2223.json"
_OPEN_2223_PATH = "eval/djev/a_read_2223.json"
_PROSE_DOC = "autonomy/9223-prose-creation.md"
_WROTE_2223_ROW = f"{_PROSE_DOC}::repo:{_WROTE_2223_PATH}"
_OPEN_2223_ROW = f"{_PROSE_DOC}::repo:{_OPEN_2223_PATH}"


def _history_body() -> str:
    """The shape `autonomy/96-djev-name-prior-probe.md` has, wrapped where it
    wraps: the creation sentence breaks line just before the backticked path, and
    the same document also tells a run to open a file it must read.
    """
    return (f"The probe ran. It wrote\n`{_WROTE_2223}` and left\n"
            "`name_prior_2026-09-24.json` untouched. "
            f"Read `{_OPEN_2223}` for the flip rates.\n")


def test_a_prose_creation_record_is_not_drift_and_a_read_instruction_is():
    """Both halves of the #2223 clause in one node: the sentence shape that red
    the live corpus produces no row, and the sentence beside it still produces
    exactly one.

    The scanner has to yield BOTH paths before the exemption is asked, because a
    rule that silenced the whole document — or a matcher that stopped resolving
    anchored paths — would make an empty assertion here read as a fix. So the
    denominator is asserted first, the way `test_the_path_check_anchored_to_a_path_
    that_really_exists` does for the corpus.
    """
    body = _history_body()
    named = _named_paths(body, None)
    assert {("repo", _WROTE_2223_PATH), ("repo", _OPEN_2223_PATH)} <= named, (
        f"the scanner stopped yielding one of the two paths this node is about: "
        f"{sorted(named)}")
    rows = _absent_refs(_PROSE_DOC, body, None)
    assert rows == {_OPEN_2223_ROW}, (
        f"a history sentence about a report the job wrote was read as drift, or a "
        f"path the document only tells a run to open went unseen: {sorted(rows)}")


@pytest.mark.parametrize("sentence", [
    "It wrote {p} and left the earlier report untouched.",
    "The run created {p} before it exited.",
    "The report was written to {p} by the probe.",
    "Each night the job generates {p} fresh.",
    "It saves {p} and prints the path.",
])
def test_a_creation_stated_in_prose_exempts_only_the_path_it_names(sentence):
    """Five wordings of the same claim, because the corpus writes it in more than
    one tense and the rule is about the statement, not about one verb form. Each
    body also names a path the doc merely tells a run to read, so a rule that
    exempted the whole document shows up here as an empty set.
    """
    body = sentence.format(p=f"`{_WROTE_2223}`") + f" The same doc reads `{_OPEN_2223}`.\n"
    rows = _absent_refs(_PROSE_DOC, body, None)
    assert rows == {_OPEN_2223_ROW}, (
        f"{sentence!r} did not exempt exactly the report it names: {sorted(rows)}")


@pytest.mark.parametrize("verb", ["Read", "Open", "Run", "Check", "Patch"])
def test_a_verb_that_is_not_a_creation_record_keeps_the_row(verb):
    """The direction the fence exists to keep biting: #85's defect was a
    instruction naming a file no run can open, and a verb that asks a reader to
    touch a path is that instruction whatever tense it wears.
    """
    rows = _absent_refs("autonomy/9223-read.md", f"{verb} `{_WROTE_2223}` first.\n",
                        None)
    assert rows == {f"autonomy/9223-read.md::repo:{_WROTE_2223_PATH}"}, (
        f"a {verb}-style reference to an absent path stopped being a violation: "
        f"{sorted(rows)}")


@pytest.mark.parametrize("site", [
    "<path>",                                     # a placeholder, not a claim
    "$(date -u +%F).json",                          # a shell substitution
    "$HOME/lloyd/eval/djev/a_created_2223.json",
    "./eval/djev/a_created_2223.json",
    "eval/djev/a_created_2223.json",     # bare relative: the scanner reads this one
])
def test_a_prose_creation_site_that_is_not_a_rooted_path_exempts_nothing(site):
    """Fail-closed, keeping `_declared_outputs`' scoping (#2157's five shapes plus
    the bare relative word, which `_named_paths` DOES read from a backtick and so
    is the case where an exemption keyed on the string rather than on a resolved
    `(tree, rel)` would lose a real drift row).
    """
    body = f"It wrote `{site}` and the report lives at {_WROTE_2223}\n"
    rows = _absent_refs("skills/2223-writer/SKILL.md", body, None)
    assert rows == {f"skills/2223-writer/SKILL.md::repo:{_WROTE_2223_PATH}"}, (
        f"a prose write site that is not a rooted path ({site!r}) exempted an "
        f"absent file: {sorted(rows)}")


def test_a_prose_creation_record_is_scoped_to_its_document_and_exact_path():
    """The two ways this exemption could quietly rot into a list nobody maintains,
    the same two #2157 pinned for flag sites.

    A second document naming the same report without creating it is drift again:
    task 96's sentence exempts nothing for any other file, so the corpus never
    grows a set of known outputs to keep current. And a sentence about the job
    writing the DIRECTORY exempts no file inside it — an exact path, because a
    parent named once would otherwise cover every report anyone files there.
    """
    other = _absent_refs("autonomy/9223-other.md",
                         f"The report lives at {_WROTE_2223}\n", None)
    assert other == {f"autonomy/9223-other.md::repo:{_WROTE_2223_PATH}"}, (
        f"a document that only names a path stopped being asked about it: "
        f"{sorted(other)}")

    parent = _absent_refs("autonomy/9223-parent.md",
                          f"It wrote `~/lloyd/eval/djev` and the report is "
                          f"{_WROTE_2223}\n", None)
    assert parent == {f"autonomy/9223-parent.md::repo:{_WROTE_2223_PATH}"}, (
        f"a creation sentence naming the parent directory exempted a file inside "
        f"it: {sorted(parent)}")


def test_the_name_prior_probe_history_sentence_makes_no_row_in_the_live_corpus():
    """The corpus half of #2223: the document that reddened three nodes here,
    measured against the live vault and this checkout, with the vault untouched.

    `autonomy/96-djev-name-prior-probe.md` records that the probe run wrote
    `eval/djev/name_prior_<date>.json`. That report is untracked in the code tree
    — the task's own Never list bars it from `git add` — and `tests/conftest.py`
    forces this suite into a `git worktree` cut from HEAD, where an untracked file
    does not exist however fresh it is on disk. So a true history sentence became
    a dead-path row, and since the report is dated it arrived again the following
    week. The item refuses deleting the history, so what changes is the node.
    """
    drift = {r for r in _unresolved() if "name_prior" in r}
    assert drift == set(), (
        f"a dated report the probe wrote is being counted as drift again: "
        f"{sorted(drift)}")

    doc = VAULT / "autonomy" / "96-djev-name-prior-probe.md"
    if doc.exists():
        body = doc.read_text(encoding="utf-8", errors="replace")
        assert re.search(r"It wrote\s+`/home/alansrobotlab/lloyd/eval/djev/"
                         r"name_prior_[0-9-]{8,10}\.json`", body), (
            "the history sentence this rule was written for has left the document, "
            "so this node is no longer crediting the exemption to anything")
        rows = _absent_refs("autonomy/96-djev-name-prior-probe.md", body, None)
        assert not [r for r in rows if "name_prior" in r], (
            f"the rule resolved elsewhere and the history sentence is a row again: "
            f"{sorted(rows)}")


# ---------------------------------------------------------------------------
# #2223 half two: a path quoted as the payload of an exception is an absence
# record, not a citation — the polarity half that tense cannot carry
# ---------------------------------------------------------------------------

_ERRORED_2223 = "~/lloyd/eval/djev/a_errored_2223.json"
_ERRORED_2223_PATH = "eval/djev/a_errored_2223.json"
_ERR_DOC = "autonomy/9223-runlog.md"
_ERR_ROW = f"{_ERR_DOC}::repo:{_ERRORED_2223_PATH}"
# The same `read` instruction this file plants elsewhere, under THIS node's label.
_ERR_OPEN_ROW = f"{_ERR_DOC}::repo:{_OPEN_2223_PATH}"


def test_a_path_named_only_as_an_exception_payload_makes_no_row():
    """`FileNotFoundError: <path>` is the fence's own verdict, already published.

    The second #2223 offender is not task 96 at all: by the time this round ran,
    `autonomy/76-queue-health-check.md` had appended a run record quoting
    `FileNotFoundError: /home/…/memory/scratchpads/2f363115-…`, and that row reddened
    the same three nodes for the same reason — a document faithfully reporting that
    a path could not be opened counted as a claim that the path should exist. A
    creation record and an absence record are both past tense and point opposite
    ways, which is why the rule keys on the speaker.

    The denominator is asserted through a sibling path the same document asks a run
    to open, so a rule that silenced the whole page cannot pass here.
    """
    body = (f"The run failed. The traceback's last line reads\n"
            f"`FileNotFoundError: {_ERRORED_2223}`\n"
            f"Read `{_OPEN_2223}` for the flip rates.\n")
    named = _named_paths(body, None)
    assert {("repo", _ERRORED_2223_PATH), ("repo", _OPEN_2223_PATH)} <= named, (
        f"the scanner stopped yielding one of the two paths this node is about: "
        f"{sorted(named)}")
    rows = _absent_refs(_ERR_DOC, body, None)
    assert rows == {_ERR_OPEN_ROW}, (
        f"a quoted FileNotFoundError naming the missing file was counted as drift, "
        f"or the read instruction beside it went unseen: {sorted(rows)}")

    quoted = (f"`FileNotFoundError: [Errno 2] No such file or directory: "
              f"'{_ERRORED_2223}'`\n")
    assert _absent_refs(_ERR_DOC, quoted, None) == set(), (
        "the errno form of the same message, which is what Python's own "
        "FileNotFoundError prints, is not recognised")


@pytest.mark.parametrize("sentence", [
    "ValueError: bad key in {p} — fix it by hand.",
    "KeyError: while editing the config at {p} the parser gave up.",
    "RuntimeError: copy the template to {p} and retry.",
])
def test_an_exception_message_that_only_mentions_a_path_keeps_the_row(sentence):
    """Fail-closed on the payload position, which is what keeps this from becoming
    "anything near the word Error is exempt".

    A message *about* something else that happens to name a path (`ValueError: bad
    key in ~/lloyd/x`) is a live reference with a dramatic wrapper — the reader is
    still told to go fix that file — and the third is an instruction wearing an
    exception's clothes. Only a message whose entire payload is the path says the
    path is missing, which is the one claim that cannot also be a citation.
    """
    body = sentence.format(p=_ERRORED_2223) + f"\nThe doc also reads `{_OPEN_2223}`.\n"
    rows = _absent_refs(_ERR_DOC, body, None)
    assert _ERR_ROW in rows, (
        f"{sentence!r} is not an absence record and lost its row: {sorted(rows)}")


@pytest.mark.parametrize("payload", [
    "$HOME/lloyd/eval/djev/a_errored_2223.json",
    "./eval/djev/a_errored_2223.json",
    "eval/djev/a_errored_2223.json",
])
def test_an_unrooted_exception_payload_exempts_nothing(payload):
    """Same scoping as every sibling rule: the exemption is keyed on a resolved,
    root-anchored `(tree, rel)`, so a payload that normalizes to nothing exempts
    nothing and the drift row survives.
    """
    body = (f"`FileNotFoundError: {payload}`\n"
            f"The report lives at {_ERRORED_2223}\n")
    rows = _absent_refs(_ERR_DOC, body, None)
    assert rows == {_ERR_ROW}, (
        f"an unrooted exception payload ({payload!r}) exempted an absent file: "
        f"{sorted(rows)}")


def test_the_run_record_row_that_red_this_file_makes_no_row_in_the_live_corpus():
    """The live-corpus half of the second offender, with the vault left alone.

    `autonomy/76-queue-health-check.md:122` is one bullet of an Activity Log the
    autonomy harness appends after every run of task 76, and the writer clips it
    mid-path with no `…` to mark it — the bullet ends `-c4f` where the file it
    quotes was `…-c4f.md`. So it cannot be reworded at the source the way #2026's
    incident note was (the next run re-appends it) and it cannot be a
    `PATH_KNOWN_UNFIXED` entry (one UUID per incident, unbounded). It is the reason
    this is a rule and not a cleanup.
    """
    drift = {r for r in _unresolved() if "scratchpads/" in r}
    assert drift == set(), (
        f"a quoted FileNotFoundError is counted as drift again: {sorted(drift)}")

    doc = VAULT / "autonomy" / "76-queue-health-check.md"
    if doc.exists():
        # The denominator again: the row this rule was written for has to still be
        # in the document, or the assertion above is crediting the exemption to
        # nothing and would pass on a corpus that stopped containing the case.
        body = doc.read_text(encoding="utf-8", errors="replace")
        assert re.search(r"FileNotFoundError:\s*/home/\S+", body), (
            "task 76's Activity Log no longer quotes a FileNotFoundError, so this "
            "node is no longer measuring the shape it exists for")
        rows = _absent_refs("autonomy/76-queue-health-check.md", body, None)
        assert not [r for r in rows if "scratchpads/" in r], (
            f"the rule resolved elsewhere and the run record is a row again: "
            f"{sorted(rows)}")


# --- #2354: a generated dangling report is ABOUT its paths, not citing them -----

_LEDGER_2354 = "autonomy/92354-referential-integrity.md"
_LISTED_2354 = "scripts/a_script_the_ledger_lists_2354.py"
_PROSED_2354 = "scripts/a_script_the_ledger_prose_names_2354.py"
_INSTRUMENT_2354 = "scripts/maintenance/referential_integrity.py"
_LISTED_ROW = f"{_LEDGER_2354}::repo:{_LISTED_2354}"
_PROSED_ROW = f"{_LEDGER_2354}::repo:{_PROSED_2354}"


def _ledger_body(listed=(), prose=(), *, keys=None, bullets=True, raw_block=None):
    """The shape `scripts/maintenance/referential_integrity.py:313-327` writes: a
    header naming the instrument that ran, one bullet per dangling cite, then the
    `ri:dangling` machine block that script appends for its own night-to-night diff.

    `listed` is what the sweep enumerated. `prose` is what the report's SENTENCES name
    — the script a reader is told to run — which stays a citation, because the rule keys
    on enumeration and not on the document as a whole. `keys` overrides the block's
    payload verbatim and `raw_block` replaces the whole block comment, so a malformed
    one can be tested; `bullets=False` drops the bullet lines so the block alone is in
    play.
    """
    lines = ["# Referential integrity — loaded memory", "",
             f"Run 2026-10-07T12:00:54Z by `{_INSTRUMENT_2354}` (#882). Report only.",
             "", "## Dangling", ""]
    if bullets:
        for i, path in enumerate(listed):
            lines.append(f"- `lloyd/USER.md:{28 + i}` → `{path}` — "
                         "stat:repo_root|vault_root|data_root|citing_dir (none exist)")
    else:
        lines.append("- none")
    lines += ["", "## Exempt", "", "- none", ""]
    for path in prose:
        lines.append(f"See `{path}` before re-running the sweep.")
    if raw_block is not None:
        lines += ["", raw_block, ""]
    else:
        payload = (keys if keys is not None
                   else [f"lloyd/USER.md:{28 + i} -> {p}"
                         for i, p in enumerate(listed)])
        lines += ["", f"<!-- ri:dangling {json.dumps(payload)} -->", ""]
    return "\n".join(lines)


def test_what_a_dangling_ledger_lists_makes_no_row_and_what_its_prose_names_does():
    """#2354 clause 2: the referential-integrity report is an enumeration of cites that
    do not resolve, so a path it lists is the path it is ABOUT, and a path its prose
    tells a reader to run is still a citation it owes.

    The denominators beside the assertion are the point of the node: `_recorded_sites`
    is a subtraction, so a fixture whose absent scripts the scanner never yielded in the
    first place turns an empty row set green while the rule does nothing — the exact
    shape #2312 was filed for. Both scripts are checked absent first, then named, then
    the row set is compared exactly, and the instrument the header cites is asserted
    present in `named` so the fixture cannot silently stop looking like the report it
    models.
    """
    for rel in (_LISTED_2354, _PROSED_2354):
        assert not (ROOT / rel).exists(), f"{rel} really exists in the checkout"
    body = _ledger_body(listed=(_LISTED_2354,), prose=(_PROSED_2354,))
    named = set(_named_paths(body, None))
    assert {("repo", _LISTED_2354), ("repo", _PROSED_2354)} <= named, (
        "the scanner yielded neither of the two absent scripts this node is about, so "
        f"an empty row set would prove nothing: {sorted(named)}")
    assert ("repo", _INSTRUMENT_2354) in named, (
        "the fixture no longer names the instrument the way the real report does, so "
        "the control below measures nothing")
    rows = _absent_refs(_LEDGER_2354, body, None)
    assert rows == {_PROSED_ROW}, (
        "a path the report enumerates should be its subject and a path its prose "
        f"names should keep its row: {sorted(rows)}")


def test_a_dangling_target_exempted_under_one_tree_does_not_claim_the_other():
    """One tree per target, the direction that fails safe.

    `~/obsidian/scripts/x` and `scripts/x` are the same string after the root, but the
    report proved only the first is gone and the checkout copy is a different file: a
    rule that exempted both would let a vault row silently excuse a skill's claim about
    a checkout path that the sweep never looked at. So the vault-spelled target is
    swallowed and the checkout-spelled one keeps its row.
    """
    rel = "scripts/a_script_the_ledger_lists_2354.py"
    assert not (VAULT / rel).exists(), f"{rel} really exists in the vault"
    assert not (ROOT / rel).exists(), f"{rel} really exists in the checkout"
    body = _ledger_body(listed=(f"~/obsidian/{rel}",), prose=(rel,))
    named = set(_named_paths(body, None))
    assert {("vault", rel), ("repo", rel)} <= named, (
        f"the scanner yielded only one spelling, so the comparison is vacuous: "
        f"{sorted(named)}")
    rows = _absent_refs(_LEDGER_2354, body, None)
    assert rows == {f"{_LEDGER_2354}::repo:{rel}"}, (
        "the vault-spelled target leaked into the checkout tree: "
        f"{sorted(rows)}")


@pytest.mark.parametrize("raw_block", [
    '<!-- ri:dangling ["lloyd/USER.md:28 -> scripts/a_2354.py"',
    '<!-- ri:dangling {"dangling": "an object where the sweep writes a list"} -->',
    '<!-- ri:dangling [unclosed, -->',
])
def test_a_ledger_whose_block_cannot_be_read_exempts_nothing(raw_block):
    """The gate is a PARSEABLE block, and it fails closed.

    A report truncated mid-write, one holding an object where the sweep writes a list,
    and one whose payload is not JSON each exempt nothing — including the well-formed
    bullets sitting above them, because a document that cannot prove it is a generated
    sweep has no proof to spend on a path. Main went red on a report that happened to be
    accurate; going green on one that is not, or on a hand-written note that only looks
    like one, would be the worse failure.
    """
    body = _ledger_body(listed=(_LISTED_2354,), prose=(_PROSED_2354,),
                        raw_block=raw_block)
    rows = _absent_refs(_LEDGER_2354, body, None)
    assert rows == {_LISTED_ROW, _PROSED_ROW}, (
        f"a block that cannot be parsed still exempted a path ({raw_block!r}): "
        f"{sorted(rows)}")


#: Each `(entry, what it yields)` pair is one line of a `ri:dangling` payload. The
#: last row is the control: the ONE spelling among these that the sweep itself could
#: have written is the one that yields a target.
@pytest.mark.parametrize("entry, expected", [
    (123, set()),
    ("no arrow separator here", set()),
    ("lloyd/USER.md:28 -> ~/lloyd/scripts/a_script_the_ledger_lists_2354.py […]",
     set()),
    ("lloyd/USER.md:28 -> obsidian/scripts/a_script_the_ledger_lists_2354.py", set()),
    ("lloyd/USER.md:28 -> ~/lloyd-data/_pipeline/owed-ledger.json", set()),
    ("lloyd/USER.md:28 -> ~/lloyd/scripts/a_script_the_ledger_lists_2354.py",
     {("repo", "scripts/a_script_the_ledger_lists_2354.py")}),
])
def test_an_entry_counts_only_when_it_names_a_scannable_target(entry, expected):
    """Inside a block that parses, an entry still has to name a target the scanner could
    have produced, and it contributes its TARGET and never its citing half.

    A non-string, a key with no ` -> ` separator, a clipped path, a root this test does
    not model (the data root), a spelling neither rooted regex accepts, and the citing
    half of every one of them each yield nothing — so a malformed report can only leave
    a row red, never excuse one. `lloyd/USER.md` sits on the left of a real dangling key
    every night and is still not exempt from naming a path it needs.

    Asserted against `_dangling_targets` and not `_absent_refs` because the claim here
    is about what one entry names. The end-to-end node through the whole row pipeline is
    `test_what_a_dangling_ledger_lists_makes_no_row_and_what_its_prose_names_does`.
    """
    body = _ledger_body(listed=(), bullets=False, keys=[entry])
    assert _dangling_targets(body) == expected, (
        f"entry {entry!r} yielded the wrong target set")


def test_a_path_the_ledger_lists_still_reads_as_drift_in_a_skill():
    """The exemption belongs to the document that enumerated the path.

    `_recorded_sites` is called one body at a time, and that scoping is the whole reason
    this is a rule and not a list: a skill that names a script the nightly sweep happens
    to have listed as dangling is still claiming a path it needs, and no report written
    about somebody else's cite can retire that claim.
    """
    body = f"See `{_LISTED_2354}` before re-running the sweep.\n"
    rows = _absent_refs("skills/2354-some-skill/SKILL.md", body, None)
    assert rows == {"skills/2354-some-skill/SKILL.md::repo:" + _LISTED_2354}, (
        f"the ledger's exemption leaked into another document: {sorted(rows)}")


def test_the_live_dangling_report_loses_its_own_rows_and_nothing_else():
    """#2354 on the document that actually red main, with a denominator that says the
    rule did work.

    The dangling report is rewritten every night, so this node asserts the shape and
    never a count: the report contributes no row at all; at least one path it enumerates
    is a path the scanner named and could not find, which is what the rule swallowed;
    and the resolver its own header cites is named but NOT swallowed, which is what
    keeps "the report is its own subject" from collapsing into "the report is exempt".
    """
    label = "autonomy/referential-integrity-latest.md"
    doc = VAULT / label
    assert doc.exists(), f"{label} is not where the nightly job writes it"
    body = doc.read_text(encoding="utf-8", errors="replace")

    def _exists(pair):
        tree, rel = pair
        return ((ROOT if tree == "repo" else VAULT) / rel).exists()

    assert _absent_refs(label, body, None) == set(), (
        "the dangling report is still being read as a set of citations")
    named = set(_named_paths(body, None))
    swallowed = _dangling_targets(body)
    assert any(p in named and not _exists(p) for p in swallowed), (
        "the rule swallowed nothing on the real report, so the assertion above proves "
        f"nothing about it; the sweep named {sorted(named)[:6]} this run")
    assert ("repo", _INSTRUMENT_2354) in named, (
        "the live report stopped naming the resolver that writes it")
    assert ("repo", _INSTRUMENT_2354) not in swallowed, (
        "the rule is swallowing the report's own prose citation of the resolver, which "
        "is silencing the document rather than reading it as a subject")
