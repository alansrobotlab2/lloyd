"""What the research skills claim, checked against the machine.

Three of the four signal paths the generator skill named did not exist. Not
subtly: `~/obsidian/pending-research/` is not a directory, gap-fill has never
written a note, and the daily notes are at `~/obsidian/memory/`, not
`~/obsidian/lloyd/memory/`. The model that ran it worked that out at runtime,
said so in its report, and nobody read the report — so the skill kept sending
every run to look in three empty places for months.

A skill is a prompt with no compiler. This file is the compiler for the parts
that are checkable: does the path exist, does the tool exist, does the number
in the doc match the code.
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
VAULT = Path.home() / "obsidian"

GENERATOR = VAULT / "skills" / "research-queue-generator" / "SKILL.md"
DEEP_DIVE = VAULT / "skills" / "deep-dive-research" / "SKILL.md"
DOC = ROOT / "architecture" / "research-pipeline.md"

#: A backticked path in a skill, `~`-anchored. Globs are resolved as "at least
#: one match", because these name a dated directory or file per run.
_PATH_RE = re.compile(r"`(~/[A-Za-z0-9_./*<>-]+)`")


def _skill_paths(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out = []
    for raw in _PATH_RE.findall(text):
        # `<last 3 days>` is prose inside a path and `YYYY-MM-DD` is a date
        # template; both mean "one of these per run", so both become a glob.
        cleaned = re.sub(r"<[^>]*>", "*", raw)
        cleaned = cleaned.replace("YYYY-MM-DD", "*")
        out.append(cleaned)
    return out


#: Roots whose contents a job writes and rewrites. A doc naming something under one
#: of these is naming an output, not a file in the checkout, so its absence is a
#: statement about the machine. All four are gitignored; `_pipeline/` and `sessions/`
#: were emptied by the 2026-09-22 tree deletion and are refilled by the jobs that
#: produce them, so a hard failure here is a red node at base in every round for a
#: path no commit broke.
RUNTIME_ROOTS = ("_pipeline/", "sessions/", "logs/", "data/")


def _is_runtime_absence(spec: str) -> bool:
    """True when `spec` names a job's output under a root that holds none yet.

    Deliberately narrow: it does not ask whether the path exists (the caller already
    knows it does not), only whether the thing named is regenerable output. A typo in
    a `scripts/` or `app/` path is still drift and still fails.
    """
    rel = spec
    for prefix in ("~/lloyd-data/", f"{Path.home()}/lloyd-data/", "~/lloyd/", f"{Path.home()}/lloyd/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return rel.startswith(RUNTIME_ROOTS)


def _resolves(spec: str) -> bool:
    # `~/lloyd-data` is the live data root however `$HOME` is set: under a gate
    # it would otherwise name the round's own, empty one (app.paths).
    from app.paths import production_data_root
    if spec.startswith("~/lloyd-data"):
        spec = str(production_data_root()) + spec[len("~/lloyd-data"):]
    p = Path(spec.replace("~", str(Path.home()), 1))
    if "*" not in spec:
        return p.exists()
    # Walk down to the first globbed segment and match from there.
    parts = p.parts
    for i, part in enumerate(parts):
        if "*" in part:
            base = Path(*parts[:i])
            pattern = str(Path(*parts[i:]))
            return base.exists() and any(base.glob(pattern))
    return p.exists()


@pytest.mark.parametrize("skill", [GENERATOR, DEEP_DIVE], ids=["generator", "deep-dive"])
def test_the_skill_exists(skill):
    assert skill.exists(), f"{skill} is missing"


@pytest.mark.parametrize("skill", [GENERATOR, DEEP_DIVE], ids=["generator", "deep-dive"])
def test_every_path_a_skill_names_resolves(skill):
    """The regression: three of four signal paths pointed at nothing."""
    missing = [spec for spec in _skill_paths(skill) if not _resolves(spec)]
    drift = [spec for spec in missing if not _is_runtime_absence(spec)]
    assert not drift, (
        f"{skill.name} names paths that do not exist: {drift}. A skill is a "
        f"prompt with no compiler; this is the compiler.")
    if missing:
        pytest.skip(
            f"{skill.name} names only regenerable output that this machine holds "
            f"none of yet: {missing}. The 2026-09-22 deletion emptied these roots "
            "and the producing jobs refill them; nothing here is a wrong claim.")


def test_the_generator_names_the_four_live_signals():
    text = GENERATOR.read_text(encoding="utf-8")
    assert "knowledge-health-" in text
    assert "session-distill" in text and "## Gaps" in text
    assert "backlog_tasks" in text
    assert "obsidian/memory/" in text
    assert "gap-fill" not in text, "gap-fill has never produced a note"


def test_neither_skill_reads_the_retired_checklist():
    """It is renamed and imported; a skill still reading it would find an
    archive header and 3,690 lines of history."""
    for skill in (GENERATOR, DEEP_DIVE):
        text = skill.read_text(encoding="utf-8")
        assert "research-queue.md" not in text.replace("research-queue-archive.md", ""), \
            f"{skill.name} still reads the retired checklist"


def test_the_retired_checklist_is_archived_and_unreferenced_by_code():
    assert not (VAULT / "lloyd" / "research-queue.md").exists()
    archive = VAULT / "lloyd" / "research-queue-archive.md"
    assert archive.exists() and "Archived" in archive.read_text(encoding="utf-8")[:400]

    # Code, not prose: several docstrings still recount what reading that file
    # cost, and that history is worth keeping. What must not survive is a
    # module that still opens it.
    for py in (ROOT / "workers").rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        live = [n for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docstrings and "research-queue.md" in n.value]
        assert not live, f"{py} still reads the retired checklist"


def test_every_tool_the_skills_call_is_advertised():
    """A skill naming a tool that does not exist is a run that fails at the
    moment it matters."""
    import agent_mcp.main as M

    real = {t.name for t in asyncio.run(M.list_tools())}
    named = set()
    for skill in (GENERATOR, DEEP_DIVE):
        text = skill.read_text(encoding="utf-8")
        named |= set(re.findall(r"\b(research_[a-z_]+|backlog_tasks|vault_recall|"
                                r"vault_write|fact_add)\b", text))
    assert named, "the skills name no tools at all — did they get rewritten?"
    assert named <= real, f"skills name tools that do not exist: {sorted(named - real)}"


def test_the_deep_dive_skill_carries_the_contract_the_worker_parses():
    text = DEEP_DIVE.read_text(encoding="utf-8")
    assert "RESULT:" in text
    for verb in ("written", "nothing_found", "duplicate"):
        assert verb in text, verb
    assert "source: deep-research" in text, "the note's frontmatter names the source"
    assert "topic_id:" in text
    assert "date +%F" not in text, (
        "the source supplies the date and the path; that bash call is why past "
        "runs produced notes misdated by days")


def test_the_deep_dive_skill_does_not_pick_its_own_topic():
    text = DEEP_DIVE.read_text(encoding="utf-8")
    assert "The topic is given to you" in text
    assert "fallback" not in text.lower()


def test_the_generator_acts_on_similar_rather_than_reading_it():
    """Without this rule the model treats `similar` as informational and
    proposes the reword anyway — which is how the old queue accumulated the
    same 82 topics 390 times."""
    text = GENERATOR.read_text(encoding="utf-8")
    assert "similar" in text
    assert "Act on `similar`" in text


def test_the_generator_checks_the_code_before_calling_something_broken():
    """On 2026-09-07 it proposed the TTS EQ and speed control as open defects.
    Both were implemented and configured."""
    text = GENERATOR.read_text(encoding="utf-8")
    assert "Grep" in text and "before" in text
    assert (ROOT / "agent-services" / "tts_shaping.py").exists(), \
        "the example the skill cites must stay true"


# ---------------------------------------------------------------------------
# Task files and config
# ---------------------------------------------------------------------------


def test_the_generator_task_has_room_to_finish():
    """900s auto-disabled it on 2026-09-04 after three consecutive timeouts."""
    import autonomy

    task = autonomy._parse_task_file(autonomy._find_task_file(65))
    assert task and task["skill_name"] == "research-queue-generator"
    assert int(task["timeout_seconds"]) >= 1500

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    pool_cap = cfg["workers"]["sources"]["scheduled-task"]["max_duration_seconds"]
    assert int(task["timeout_seconds"]) < pool_cap - 30, (
        "the task's own timer must fire before the pool's, or the pool cancels "
        "it and no run record is written at all")


def test_the_generator_description_is_what_the_model_actually_reads():
    """`_build_task_prompt` renders the skill and the `description` field, and
    never the markdown body — so a stale description is a stale prompt."""
    import autonomy

    task = autonomy._parse_task_file(autonomy._find_task_file(65))
    desc = task["description"]
    assert "research_propose" in desc or "registry" in desc
    assert "research-queue.md" not in desc


def test_the_deep_dive_task_is_retired_not_still_scheduled():
    """It runs as the `deep-research` worker source now. Two schedulers for one
    skill would research two topics a day and record one."""
    import autonomy

    assert autonomy._find_task_file(52) is None, "#52 is still dispatchable"
    archived = VAULT / "autonomy" / "_archived" / "52-deep-dive-research.md"
    assert archived.exists()
    task = autonomy._parse_task_file(archived)
    assert task["status"] == "paused"
    assert "archived_reason" in task


# ---------------------------------------------------------------------------
# The task-body contract (#951)
# ---------------------------------------------------------------------------
#
# `_build_task_prompt` renders the silent-run hint, the task header, the
# `skill_name` SKILL.md and the front-matter `description`. It never reads the
# markdown body, so the body is documentation plus the machine-written activity
# log — not an instruction channel. That is the contract #951 writes down, and
# the shape the system had already chosen in practice (`86ae9da` moved task #86's
# entire procedure into its front-matter `description` for exactly this reason).
#
# What the body used to be instead: #47 ordered its run to work the knowledge
# graph with tools its prompt never named, and that phase sat there unread until
# #463 deleted it on 2026-09-23 (`ad7a0d6f`). Three things pin the decision here:
# every body says what it is, the sentence is true (the built prompt really does
# exclude the body), and nothing in a body may order the run to use a tool the
# prompt does not name.

TASK_DIR = VAULT / "autonomy"

#: Where an instruction region stops. Same convention
#: `tests/test_legacy_alias_export_fate.py` already uses for the same question —
#: everything after `## Activity Log` is history the scheduler never executes,
#: and a `- 2026-…T…: …` bullet is a run report, so an honest log line cannot
#: trip the guard. Two pins asking "what does this body tell its run to do?" must
#: not answer it two different ways.
_ACTIVITY_HEADING_RE = re.compile(r"^##\s+Activity", re.I | re.MULTILINE)
_DATED_BULLET_RE = re.compile(r"^-\s+\d{4}-\d{2}-\d{2}T")
#: Names the guard has to be able to see for its verdict to mean anything. Every
#: one is served by a local module (vault, fact store, backlog), so none of them
#: vanishes when an external bridge is down — which is why `_lloyd_tool_names`
#: floors on these names rather than on a count, and why the fixture test below
#: reuses them as its fixture tools instead of inventing names of its own.
_GUARD_MUST_SEE = frozenset({"vault_write", "vault_recall", "fact_add",
                             "fact_relate", "backlog_write_task"})


def _task_files(task_dir: Path = TASK_DIR) -> list[Path]:
    """The task files the scheduler dispatches: numbered files, not `_config.md`
    or the reports that share the directory."""
    return sorted(task_dir.glob("[0-9]*-*.md"))


def _body_of(path: Path) -> str:
    """Everything after the front matter, the way `_parse_task_file` sees it."""
    parts = path.read_text(encoding="utf-8").split("---\n", 2)
    return parts[2] if len(parts) > 2 else ""


def _instruction_region(body: str) -> str:
    """The part of a body that registers work: before the activity log, minus its
    dated run-report bullets.

    `MULTILINE` on the heading regex is load-bearing and was this helper's first
    bug: `search` over a whole body with a bare `^` matches only at offset 0, and
    no task body begins with its activity heading, so without the flag the cut
    never fires and every machine-written run note is graded as an instruction.
    Both cuts are therefore pinned by name in
    `test_the_body_tool_guard_fires_on_an_undelivered_tool`, which is what stops
    either one rotting into a no-op.
    """
    m = _ACTIVITY_HEADING_RE.search(body)
    region = body[:m.start()] if m else body
    return "\n".join(ln for ln in region.splitlines()
                     if not _DATED_BULLET_RE.match(ln))


def _lloyd_tool_names() -> frozenset[str]:
    """Every tool a run could possibly be told about: the central annotation
    table plus every tool the MCP server actually advertises.

    Both, because the table alone is not the tool surface. Measured at `e4ef79c4`
    it holds 122 of the 147 names `list_tools()` advertises, and the 26 it omits are
    disproportionately writers (`fact_add`, `memory_add`, `backlog_write_task`,
    `research_propose`, the whole email-reply family) — precisely the actions a task
    body would plausibly order, which is the #47 failure with a different verb. The
    table is still read because the clause names it and because its sets are a
    literal a reader can check.

    The counts are NOT stable across environments, which is why the floor below is
    a set of names and not a number: the same call inside a gate's round home
    returned 107 advertised names and a 132-name union, because the Thunderbird
    bridge is unreachable there and the mail, calendar, contacts and task tools are
    simply absent from the advertised surface (the import banner says so). A count
    assertion passed at 132 while 15 tools had vanished; `_GUARD_MUST_SEE` does not
    have that hole.

    The public UPPER_CASE frozensets are taken reflectively so a table added later
    is covered without anyone editing this; the two private ones hold prefixes and
    extras, not names.
    """
    import asyncio

    import agent_mcp.annotations as A
    import agent_mcp.main as M

    tables = {k: v for k, v in vars(A).items()
              if k.isupper() and not k.startswith("_")
              and isinstance(v, (frozenset, set))}
    names: set[str] = set()
    for value in tables.values():
        names |= set(value)
    annotated = len(names)
    names |= {t.name for t in asyncio.run(M.list_tools())}
    # A universe that came back empty or half-built would make the guard below
    # pass by matching nothing, which is the one outcome worse than a false alarm.
    # The real control is the NAME FLOOR, not the counts: the counts are a
    # property of the environment (see the docstring), so they were satisfied at
    # 132 in a gate's round home with 15 tools already missing from the surface.
    # `_GUARD_MUST_SEE` is a property of the guard.
    assert len(tables) >= 5, f"only {sorted(tables)} annotation tables found"
    assert annotated > 100, f"only {annotated} annotated names — the scan broke"
    assert len(names) >= annotated, "list_tools returned nothing new or broke"
    unseen = sorted(_GUARD_MUST_SEE - names)
    assert not unseen, \
        f"the tool universe is missing {unseen}; the guard below would silently "  \
        "stop seeing those actions — check that agent_mcp.main.list_tools() ran"
    return frozenset(names)


def _undelivered_tool_names(prompt: str, region: str, tools: frozenset[str]) -> list[str]:
    """Tool names the instruction region backticks that the prompt never says."""
    named = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", region))
    return sorted(n for n in named if n in tools and n not in prompt)


def _prompt_and_region(path: Path) -> tuple[str, str]:
    """Build this task's real prompt with the real loaders, and take the region
    of its body that would be ordering work."""
    import autonomy

    task = autonomy._parse_task_file(path)
    assert task, f"{path.name} does not parse — the guard cannot read it"
    skill = autonomy._load_skill_content(task.get("skill_name") or "")
    # None means the skill did not resolve, and `run_task` skips such a task;
    # treating it as empty is the conservative reading, since it only ever makes
    # the delivered prompt smaller and so can only add offenders, never hide one.
    return autonomy._build_task_prompt(task, skill or ""), _instruction_region(
        task.get("body") or "")


def test_every_task_body_says_it_is_not_delivered_to_the_worker():
    """#951 clause 1: the body is documentation, and it says so, in one line, at
    the top of every task file — and the line is not a lie, because the prompt
    the same loaders build still excludes it.

    The second half is the fork point. If a future round ever injects bodies into
    `_build_task_prompt` (the other branch #951 considered, and the one only a
    person can take, since it re-sends ~40x the current instruction channel every
    tick), this fails on every file at once, which is exactly when the sentence on
    all of them has to be rewritten.
    """
    from agent_mcp._shared import TASK_BODY_CONTRACT

    files = _task_files()
    assert files, f"{TASK_DIR} matches no task files — the pin would pass vacuously"

    missing = []
    for path in files:
        body = _body_of(path)
        first = next((ln for ln in body.splitlines() if ln.strip()), "")
        if first.strip() != TASK_BODY_CONTRACT:
            missing.append(path.name)
    assert not missing, (
        "task bodies that do not open by saying they are not delivered to the "
        f"worker: {missing}. Add `TASK_BODY_CONTRACT` as the body's first line — "
        "a file created after it landed has it already (`with_body_contract`), so "
        "a failure here is either an edit that removed it or the line drifted from "
        "`agent_mcp/_shared.py`.")

    lied = [path.name for path in files
            if TASK_BODY_CONTRACT in _prompt_and_region(path)[0]]
    assert not lied, (
        f"{lied}: the body says it is not delivered to the worker, but the prompt "
        "built for it now contains that line — `_build_task_prompt` started "
        "carrying bodies, so every task file's first line is false and the "
        "instruction/documentation split has to be re-decided, not re-worded.")


def test_no_task_body_orders_a_tool_its_prompt_never_names():
    """#951 clause 2: the one way a body can still reach a run is by naming a
    tool, so no body may name one the built prompt does not name either.

    This is the guard that did not exist when #47 told its run to call
    contradiction-resolution tools `_build_task_prompt` never delivered. It fires
    on the corpus today at zero: `fact_resolve` and `fact_invalidate` left #47's
    body with the phase #463 deleted, and the four tool mentions that remain
    anywhere in the instruction regions — `fact_add` in #54 and `vault_recall` in
    #54, #81 and #82 — are all names their own skill carries into the prompt.
    A body that orders a tool NOT in its prompt is dead text to the run and a
    lie to the reader, which is the whole failure mode of the item.
    """
    tools = _lloyd_tool_names()
    offenders = []
    for path in _task_files():
        prompt, region = _prompt_and_region(path)
        missing = _undelivered_tool_names(prompt, region, tools)
        if missing:
            offenders.append(f"{path.name}: {missing}")
    assert not offenders, (
        "task bodies ordering a tool their own prompt never names: " + "; ".join(offenders)
        + " — either name the tool in that task's front-matter `description` or in "
        "its `skill_name` SKILL.md (the only two channels a run receives), or take "
        "the instruction out of the body.")


def test_the_body_tool_guard_fires_on_an_undelivered_tool(tmp_path):
    """A corpus guard that currently finds nothing has to show it can find
    something, or it is a green light and not a check.

    Three tools, three roles, so each moving part of the guard is pinned by a name
    that only that part can put into (or keep out of) the verdict:

      * `vault_write` in the body's prose — must be CAUGHT. Remove the membership
        check in `_undelivered_tool_names` and the other two fire as well, so the
        check cannot silently stop checking.
      * `fact_relate` inside a dated run-report bullet — must be EXCLUDED. Drop
        `_DATED_BULLET_RE` and it lands in the verdict; the filter is then pinned
        by a name that exists nowhere else in the body.
      * `backlog_write_task` after the `## Activity Log` heading — must be
        EXCLUDED. Lose `re.MULTILINE` on the heading regex and `search` matches
        only at offset 0, the cut never fires, and this name appears too.

    The second file is the fix a real offender gets: name the tool in the
    front-matter `description`, which is one of the two channels the prompt is
    built from, and the same body passes.
    """
    # Three tools with three jobs, each a real Lloyd tool so the guard's membership
    # check is what decides the verdict rather than a made-up string.
    prose_tool, bullet_tool, log_tool = "vault_write", "fact_relate", "backlog_write_task"
    tools = _lloyd_tool_names()
    for fixture_tool in (prose_tool, bullet_tool, log_tool):
        assert fixture_tool in tools, \
            f"fixture tool {fixture_tool} left the universe; this pin is vacuous"

    offender = tmp_path / "999-offender.md"
    offender.write_text(
        "---\nid: 999\nname: Offender\ndescription: Summarise the notes.\n"
        "skill_name: no-such-skill-for-this-test\n---\n\n"
        f"# Offender\n\nThe run calls `{prose_tool}` on every note.\n\n"
        f"- 2026-01-02T00:00:00Z: relinked sources with `{bullet_tool}`\n\n"
        "## Activity Log\n"
        f"- 2026-01-01T00:00:00Z: ran and filed a note\n"
        f"- undated log line naming `{log_tool}` by hand\n",
        encoding="utf-8")
    fixed = tmp_path / "998-fixed.md"
    fixed.write_text(
        "---\nid: 998\nname: Fixed\n"
        f"description: Summarise the notes, then write each one with `{prose_tool}`.\n"
        "skill_name: no-such-skill-for-this-test\n---\n\n"
        f"# Fixed\n\nSummarise, then write the notes with `{prose_tool}`.\n",
        encoding="utf-8")

    region = _instruction_region(_body_of(offender))
    for excluded, why in ((bullet_tool, "the dated-bullet filter"),
                          (log_tool, "the activity-heading cut")):
        assert f"`{excluded}`" not in region, \
            f"{why} stopped excluding a run report: {excluded} is in the region"
    assert _undelivered_tool_names(*_prompt_and_region(offender), tools=tools) \
        == [prose_tool]
    assert _undelivered_tool_names(*_prompt_and_region(fixed), tools=tools) == []


def _drive_task_write_route(router_module, payload: dict):
    """POST to `/api/autonomy/task-write` the way the Mission Control autonomy page
    does — through the async route handler, not the writer underneath it.

    On this path the handler's whole contract with its caller is
    `await request.json()`, so stubbing that one method leaves everything after the
    payload in production code: the id allocation, the create branch's `"body": ""`
    that #951 names, the update branch's key overlay, and `_autonomy_write_file`.
    Calling the writer directly would test the function and leave the route — the
    surface the UI actually stands on — unverified.
    """
    import asyncio

    class _RequestBodyOnly:
        def __init__(self, payload):
            self._payload = payload

        async def json(self):
            return self._payload

    return asyncio.run(
        router_module.autonomy_task_write(_RequestBodyOnly(payload)))


def test_both_task_writers_stamp_the_body_contract(tmp_path, monkeypatch):
    """The pin above reads a directory that two live routes can create files in,
    so it holds only while a new task cannot arrive without the line — which is
    `with_body_contract`, called by both writers rather than by the create branch
    alone (`_handle_write` and `POST /api/autonomy/task-write` each put a task
    file on disk, and the Mission Control autonomy page uses the second).

    Both seams, through the real functions, into a temp dir: an MCP create, an
    HTTP-route write, and a re-write of the file the create made (the update path
    re-serialises a body that already carries the line, so the stamp must stay
    idempotent — a second copy on 32 files is noise a person would delete).
    """
    from agent_mcp import autonomy as MCP
    from agent_mcp._shared import TASK_BODY_CONTRACT
    from app.routers import autonomy as ROUTER

    mcp_dir = tmp_path / "mcp"
    http_dir = tmp_path / "http"
    monkeypatch.setattr(MCP, "AUTONOMY_DIR", mcp_dir)
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", http_dir)

    MCP._handle_write({"name": "Stamped By MCP", "description": "d",
                       "skill_name": "some-skill"})
    created = _task_files(mcp_dir)
    assert len(created) == 1, f"MCP create wrote {created}"
    assert _body_of(created[0]).lstrip("\n").startswith(TASK_BODY_CONTRACT), \
        _body_of(created[0])[:120]

    # Driven as the UI drives it: a POST body in, a file on disk out. The create
    # branch of the route is where #951's named `body: ""` default actually lives,
    # so a call straight into `_autonomy_write_file` would have skipped the exact
    # line the item says creates an unstamped file.
    written = http_dir / "1-board-made.md"
    _drive_task_write_route(ROUTER, {"name": "Board made",
                                     "skill_name": "some-skill"})
    assert written.exists(), sorted(p.name for p in http_dir.iterdir())
    assert _body_of(written).lstrip("\n").startswith(TASK_BODY_CONTRACT), \
        _body_of(written)[:160]

    # And the update branch: an edit is the proof it rewrote the file at all, and
    # the stamp surviving that rewrite (the branch reads the body back off disk) is
    # the point — a second write must not emit a second copy.
    _drive_task_write_route(ROUTER, {"id": 1, "description": "edited by the board"})
    text = written.read_text(encoding="utf-8")
    assert "edited by the board" in text, \
        "the route's update path did not rewrite the file"
    assert _body_of(written).count(TASK_BODY_CONTRACT) == 1, \
        f"the route's update path duplicated the stamp: {_body_of(written)[:200]}"

    MCP._handle_write({"id": 1, "activity_note": "second write"})
    again = _body_of(created[0])
    assert again.count(TASK_BODY_CONTRACT) == 1, again[:200]
    assert "second write" in again, "the update path stopped appending its note"


def test_the_scheduler_writers_keep_the_contract_when_they_rewrite_a_task(
        tmp_path, monkeypatch):
    """The two writers above stamp, and the scheduler's two re-writers neither
    stamp nor strip — they paste the body back verbatim, and this pins that they
    keep doing it.

    `autonomy._update_task_field` (`autonomy.py:189`) and
    `autonomy._append_activity_log` (`autonomy.py:203`) are by volume the most
    frequent writers of these 32 files: every dispatched run writes its status,
    failure count and activity note through them, and they are not the MCP tool
    or the HTTP route, so they never reach `with_body_contract`. They survive
    today only because each re-emits the front matter and reattaches `parts[2]`
    untouched — a refactor that rebuilt the body from the parsed dict instead
    (`_parse_task_file` stores it at `fm["body"]`, `autonomy.py:151`) would drop
    the line from all 32 files in one scheduler tick, and the corpus pin above
    would then blame whoever's round happened to run next.

    So the assertion is the invariant, not the mechanism: after a status write
    and two appended run notes, the contract is still the body's first non-blank
    line and is still present exactly once. A re-stamping rewriter fails the
    count; a body-rebuilding one fails the first line.
    """
    import autonomy
    from agent_mcp._shared import TASK_BODY_CONTRACT

    task_dir = tmp_path / "autonomy"
    task_dir.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", task_dir)
    path = task_dir / "997-scheduler-rewrite.md"
    path.write_text(
        "---\nid: 997\nname: Scheduler rewrite\nstatus: up_next\n"
        "description: Pinned by #951.\n---\n\n"
        f"{TASK_BODY_CONTRACT}\n\n# Scheduler rewrite\n\nProse.\n",
        encoding="utf-8")

    autonomy._update_task_field(997, status="paused")
    autonomy._append_activity_log(997, "first run")
    autonomy._append_activity_log(997, "second run")

    body = _body_of(path)
    first = next((ln for ln in body.splitlines() if ln.strip()), "")
    assert first.strip() == TASK_BODY_CONTRACT, (
        f"a scheduler re-write stopped preserving the body verbatim: {body[:160]}")
    assert body.count(TASK_BODY_CONTRACT) == 1, (
        f"a scheduler re-write duplicated the contract line: {body[:200]}")
    assert "second run" in body, "the activity writer stopped appending its note"
    assert "status: paused" in path.read_text(encoding="utf-8").split("---\n")[1], (
        "and the status writer stopped writing the field it was called with, "
        "which would mean this pin asserted nothing about a real re-write")


def test_the_retired_source_is_gone_from_config_and_the_registry():
    import workers.sources as sources

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    names = cfg["workers"]["sources"]
    assert "domain-research" not in names
    assert "deep-research" in names
    assert "domain-research" not in sources.SOURCE_REGISTRY
    assert "deep-research" in sources.SOURCE_REGISTRY


def test_the_promotion_default_for_the_old_staging_dir_survives():
    """142 `domain-research` notes are still on disk under pending-research/,
    and the Review tab promotes them by directory name."""
    import app.routers.workers as W

    staged = ROOT / "_pipeline" / "vault-derived" / "pending-research" / "domain-research"
    if staged.exists():
        assert "domain-research" in W._DEFAULT_DEST, (
            "removing this makes the leftover notes unpromotable")


# ---------------------------------------------------------------------------
# The architecture doc
# ---------------------------------------------------------------------------


def test_the_doc_exists_and_keeps_its_numbers():
    assert DOC.exists()
    text = DOC.read_text(encoding="utf-8")
    for claim in ("314", "2,839", "research.db"):
        assert claim in text, claim


def test_the_doc_names_the_states_the_store_has():
    from app import research_store

    text = DOC.read_text(encoding="utf-8")
    for status in research_store.STATUSES:
        assert status in text, f"{status} is undocumented"


def test_the_written_row_names_finish_as_the_verifier():
    """§2's `written` row claimed the disk check lives in the worker and *not*
    in `finish` (#1276). It was true when written; #1276 moved the check into
    the store, so the sentence that says it is absent is now the defect — the
    next reader would trust it and go looking for the hole in the wrong place.

    Pinned in both directions: the row must name `finish` as the verifier, and
    no sentence may still assert `finish` does not check disk.
    """
    from app import research_store

    text = DOC.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines()
               if line.startswith("| `written` |"))
    assert "`finish`" in row, "the row must name the method that verifies"
    assert "_require_real_note" in row, "and the helper that does it"
    assert str(research_store.MIN_NOTE_BYTES) in row, (
        "the byte floor the row quotes must be the one the code enforces")
    for line in text.splitlines():
        if re.search(r"finish`? itself does not", line) or \
           re.search(r"not in `finish`", line):
            raise AssertionError(
                f"a sentence still says finish does no disk check: {line.strip()[:100]}")


# ---------------------------------------------------------------------------
# Worker-source docstrings (#705)
# ---------------------------------------------------------------------------

WORKER_SOURCES = ROOT / "workers" / "sources"

#: The staging root as a checkout-relative path, spelled the way a docstring
#: spells it. Derived from `app.paths` rather than written as a literal so the
#: doc can never drift from the constant the writer uses.
STAGING_CHECKOUT_REL = "lloyd-data/_pipeline/vault-derived/pending-research"

#: A `~`-anchored path in a worker-source docstring, backticked or not.
#: `_PATH_RE` above cannot serve here for two reasons: it requires a backtick
#: on each side, and these modules name paths in RST double-backticks and in
#: bare prose alike — `_common.py` carried `~/obsidian/pending-research/` in
#: bare prose, which is the exact string that poisoned #522's acceptance
#: check; and its class has no braces, so
#: `…/pending-research/{source}/{yyyy-mm-dd}/` would be cut at the first brace
#: and the parent directory would resolve for the wrong reason.
_DOC_PATH_RE = re.compile(r"~/[A-Za-z0-9_./*<>{}-]+")

#: A staging leaf named without its root: `pending-research/gaps/`. Each of the
#: three staging sources names one, and #705 found all three wrong —
#: `distill/`, `gaps/` and `bench/` against the real `session-distill/`,
#: `gap-fill/` and `bench-mine/`. A guard that compiles only the root passes on
#: every one of those, which is why the leaf is extracted separately.
_STAGING_LEAF_RE = re.compile(r"pending-research/(?P<leaf>\{[^}]+\}|[A-Za-z0-9_.-]+)")

#: Prose punctuation glued to the end of a path is not part of the path.
_PATH_TRAILING = ".,;:`'\" "


def _doc_claims(py: Path) -> dict:
    """What one module's OWN docstring asserts about the filesystem.

    Only the module docstring: that is the surface #705 is about — the text a
    triage or selfmod run reads to decide where to go looking. `NAME` comes
    from the source text because it is the leaf `write_staging_note` fixes.
    """
    src = py.read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or ""
    name = re.search(r'^NAME\s*=\s*["\']([^"\']+)["\']', src, re.M)
    return {
        "name": name.group(1) if name else None,
        "paths": [p.rstrip(_PATH_TRAILING) for p in _DOC_PATH_RE.findall(doc)],
        "leaves": _STAGING_LEAF_RE.findall(doc),
    }


def _doc_resolves(spec: str) -> bool:
    """Resolve a docstring path; `{yyyy-mm-dd}` and `<…>` mean "at least one".

    The same rule `_skill_paths` applies to a skill's `<last 3 days>`, and it
    has to be "at least one" rather than "exists": these are dated directories,
    written one per run, so no literal path is ever the whole claim.
    """
    cleaned = re.sub(r"\{[^}]*\}", "*", spec)
    cleaned = re.sub(r"<[^>]*>", "*", cleaned).rstrip("/")
    return _resolves(cleaned)


def test_every_path_a_worker_source_docstring_names_resolves():
    """The regression: `~/obsidian/pending-research/` is not a directory.

    Every knowledge-acquisition source documented its output landing there, so
    a run that went to look got `No such file or directory` and wrote it down
    as proof nothing had been staged. #522's triage turned that into an
    acceptance clause — `≥6 staged bench task files exist under
    ~/obsidian/pending-research/bench/{yyyy-mm-dd}/` — unsatisfiable by any
    diff, because no source in the tree writes there, and three run records
    under `autonomy-runs/65/` (`run_65_20260901_174920`, `run_65_20260906_181631`,
    `run_65_20260907_221114`) each name the dead path as a finding nobody read.
    The root the code uses is `app.paths.VAULT_PENDING_RESEARCH_DIR`.

    Four kinds of failure, each with a denominator beside it so a guard that
    has stopped matching cannot report clean on an empty list:

      * the literal string `obsidian/pending-research` in any worker source,
        over the whole module and not just the docstring, which is what the
        item's own grep looks for and what nothing below can see;
      * every `~/…` path must exist, and sit under the root the code uses — a
        wrong ROOT;
      * a bare `pending-research/<leaf>/` must be that module's own `NAME` — a
        wrong LEAF, which a root-only check waves through;
      * both extractors must actually extract, one from prose that carries no
        backticks and one from a leaf that carries no root.
    """
    # First the string itself, over the WHOLE module source rather than the
    # docstring, because that is the check #705's acceptance is written as
    # (`grep -rn "obsidian/pending-research" workers/sources/*.py` is empty).
    # Everything below is anchored on `~/`, so a mention that drops the tilde
    # — `under obsidian/pending-research/` — would sail past every path
    # assertion in this file while leaving the grep red.
    scanned = sorted(WORKER_SOURCES.glob("*.py"))
    assert scanned, f"no worker sources found under {WORKER_SOURCES}"
    dead_root = sorted(p.name for p in scanned
                       if "obsidian/pending-research" in p.read_text(encoding="utf-8"))
    assert not dead_root, (
        f"these worker sources still name the vault copy of `pending-research/`, "
        f"which is not a directory and never was ({dead_root} of "
        f"{len(scanned)} files scanned). Any run that navigates by it gets "
        f"`No such file or directory` and reads that as an empty staging area.")

    claims = {p.name: _doc_claims(p) for p in scanned}
    speaking = {m: c for m, c in claims.items() if c["paths"] or c["leaves"]}
    assert len(speaking) >= 3, (
        f"the guard extracted path claims from {len(speaking)} of "
        f"{len(claims)} worker-source docstrings ({sorted(speaking)}); it must "
        f"find at least 3, or an extractor regression reads as a clean tree")

    missing = sorted({(m, spec) for m, c in speaking.items() for spec in c["paths"]
                      if not _doc_resolves(spec)})
    drift = [(m, spec) for m, spec in missing if not _is_runtime_absence(spec)]
    assert not drift, (
        f"worker-source docstrings name paths that do not exist: {drift}. A "
        f"module docstring is the map a triage run navigates by, and one wrong "
        f"path cost #522 a whole unsatisfiable acceptance clause.")
    if missing:
        pytest.skip(
            f"the only unresolved docstring paths name regenerable output this "
            f"machine holds none of yet: {missing}")

    from app.paths import DATA_ROOT, VAULT_PENDING_RESEARCH_DIR
    import app.routers.workers as W

    assert W.PENDING_ROOT == VAULT_PENDING_RESEARCH_DIR, (
        "the Review tab lists a different root than app.paths names, so the "
        "root below is not the surface a human promotes from")
    root_rel = VAULT_PENDING_RESEARCH_DIR.relative_to(DATA_ROOT)
    assert f"lloyd-data/{root_rel.as_posix()}" == STAGING_CHECKOUT_REL, (
        f"app.paths moved the staging root to {root_rel}; the docstrings and "
        f"this guard's spelling have to move with it")

    off_root = []
    for m, c in speaking.items():
        for spec in c["paths"]:
            if "pending-research" not in spec:
                continue
            parts = [p for p in spec[len("~/"):].split("/") if p]
            lead = tuple(parts[1:1 + len(root_rel.parts)])
            if parts[:1] != ["lloyd-data"] or lead != root_rel.parts:
                off_root.append((m, spec, f"~/{STAGING_CHECKOUT_REL}"))
    assert not off_root, (
        f"these docstrings name a pending-research root that is not "
        f"app.paths.VAULT_PENDING_RESEARCH_DIR (data-root-relative "
        f"{root_rel}): {off_root}")

    # Two non-vacuity pins for the extractors themselves, because each half of
    # #705 is a half a narrower guard cannot see: the poisoned line in
    # `_common.py` was bare prose (a backtick-requiring regex like `_PATH_RE`
    # extracts nothing from it), and `distill/`, `gaps/`, `bench/` were leaves,
    # which a root-only check waves through. A guard that quietly stops
    # matching one of these reports green on an empty list.
    assert f"~/{STAGING_CHECKOUT_REL}/{{source}}/{{yyyy-mm-dd}}/" in _doc_claims(
        WORKER_SOURCES / "_common.py")["paths"], (
        "the extractor only matches backticked paths, and the line that "
        "poisoned #522's acceptance check was bare prose — which is why this "
        "guard does not reuse `_PATH_RE`")
    staging = {m: c["name"] for m, c in speaking.items() if c["name"] and c["leaves"]}
    assert len(staging) >= 3, (
        f"only {len(staging)} worker-source docstrings named a "
        f"pending-research leaf ({sorted(staging)}); the three staging sources "
        f"all do, so the leaf extractor has stopped matching and a wrong leaf "
        f"would sail through while the root check stayed green")

    wrong_leaf = []
    for m, c in speaking.items():
        for leaf in c["leaves"]:
            if leaf.startswith("{"):
                continue  # a template, not a claim about one directory
            if leaf != c["name"]:
                wrong_leaf.append((m, leaf, c["name"]))
    assert not wrong_leaf, (
        f"these docstrings stage under a leaf that is not the module's own "
        f"NAME, while `write_staging_note` fixes the directory to NAME: "
        f"{wrong_leaf}. `distill/`, `gaps/` and `bench/` were all of these.")

    step3 = [ln for ln in ast.get_docstring(
        ast.parse((WORKER_SOURCES / "_common.py").read_text(encoding="utf-8"))
    ).splitlines() if "lands under" in ln]
    assert len(step3) == 1, f"step 3 of the shared pattern is not stated once: {step3}"
    assert f" ~/{STAGING_CHECKOUT_REL}/{{source}}/{{yyyy-mm-dd}}/ " in f" {step3[0]} ", (
        f"step 3 must name the checkout's own staging root with both date and "
        f"source templates, i.e. `~/{STAGING_CHECKOUT_REL}/"
        f"{{source}}/{{yyyy-mm-dd}}/`, got {step3[0]!r}")


def test_the_staging_leaf_is_the_directory_a_note_actually_lands_in(tmp_path, monkeypatch):
    """The leaf assertion above compares a docstring to a constant. This one
    compares the same promise to the only writer that makes the directory.

    Pointing `STAGING_ROOT` at a tmp dir and calling `write_staging_note`
    proves the layout the three docstrings now claim — `<root>/<NAME>/<date>/`
    — is what production code produces, and so is what
    `GET /api/workers/pending` reads back as the source name
    (`src.relative_to(PENDING_ROOT).parts[0]`). Without it `leaf == NAME`
    would only be two files agreeing with each other.
    """
    import workers.sources._common as C
    import workers.sources.bench_mine as bench_mine
    import workers.sources.gap_fill as gap_fill
    import workers.sources.session_distill as session_distill

    monkeypatch.setattr(C, "STAGING_ROOT", tmp_path)
    for mod in (bench_mine, gap_fill, session_distill):
        note = C.write_staging_note(source=mod.NAME, slug="probe", body="body")
        rel = note.relative_to(tmp_path)
        assert len(rel.parts) == 3, f"{mod.__name__}: expected root/NAME/date/note.md, got {rel}"
        assert rel.parts[0] == mod.NAME, f"{mod.__name__}: leaf is {rel.parts[0]}, not NAME"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", rel.parts[1]), (
            f"{mod.__name__}: the date directory is {rel.parts[1]!r}, not the "
            f"{{yyyy-mm-dd}} the docstrings promise")
        assert rel.parts[2] == f"{note.stem.rsplit('-', 1)[0]}-probe.md"
        assert note.read_text(encoding="utf-8").startswith("---\n"), (
            "a staged note without frontmatter is unpromotable: the Review tab "
            "reads review_status and source out of it")


def test_gap_fill_documents_the_staging_step_not_a_fact_write_it_never_makes():
    """#705's second half, and it sits two lines under the wrong path.

    The docstring said the handler "and (at high confidence) updates the fact".
    `gap_fill.execute` researches, calls `write_staging_note`, and returns: no
    path through that module writes a fact, and the whole point of the staging
    step is that a human promotes the note. A reader who believed the sentence
    would look for a confidence threshold that does not exist, and a reader of
    the facts tree would look for a `resolved_at` this source never sets.
    """
    src = (WORKER_SOURCES / "gap_fill.py").read_text(encoding="utf-8")
    doc = (ast.get_docstring(ast.parse(src)) or "").lower()

    for claim in ("updates the fact", "update the fact", "writes the fact",
                  "wrote the fact"):
        assert claim not in doc, f"gap_fill's docstring still promises a fact write: {claim!r}"
    assert "promot" in doc and "human" in doc, (
        "the docstring must say where the resolution note actually goes: "
        "staged under the source's own leaf for a human to promote")

    # The prose is only half of it: the claim fails again the moment the module
    # grows a fact writer, so pin the behaviour side too.
    for writer in ("fact_add", "fact_invalidate", "kg_store", "remember("):
        assert writer not in src, (
            f"gap_fill now uses {writer}; then the docstring is allowed to "
            f"mention a fact write again, and this test is the one to change")
