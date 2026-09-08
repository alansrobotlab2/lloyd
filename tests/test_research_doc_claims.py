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


def _resolves(spec: str) -> bool:
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
    assert not missing, (
        f"{skill.name} names paths that do not exist: {missing}. A skill is a "
        f"prompt with no compiler; this is the compiler.")


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
    import ast

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
