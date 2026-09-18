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
    "file_glob", "file_grep", "tag_search", "tag_explore",
    "pipeline_dispatch", "chat_send", "sessions_send",
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
RUNTIME_SUFFIXES = (".db", ".sqlite", ".jsonl", ".log", ".csv", ".lock", ".pid")

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


def _unresolved() -> set[str]:
    """`<label>::<tree>:<path>` for every named path that is not on disk."""
    bad: set[str] = set()
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
                bad.add(f"{label}::{tree}:{rel}")
    return bad


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
    """
    drift = _unresolved() - PATH_KNOWN_UNFIXED
    assert drift == set(), (
        "skills or autonomy tasks name paths that are not in this checkout: "
        f"{sorted(drift)}. Either put the file in the repo, point the doc at "
        "where it really lives, or — only for pre-existing drift this round did "
        "not touch — add it to PATH_KNOWN_UNFIXED with a reason."
    )


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

