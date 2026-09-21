"""One walker defines "a live skill", and every human-facing surface uses it (#1294).

Before this change five code paths each did their own `iterdir()` over the skill
roots and only two of them honoured the quarantine rule. Measured on the live tree
at 2026-09-21T11:20Z, on the commit this round was cut from, the same vault printed
five numbers:

| surface | count |
|---|---|
| `prompt_builder._load_skills_index()` (what the model is told exists) | **189** |
| `agent_mcp.skills._iter_skills()` (what `skills_search` returns) | **189** |
| `GET /api/skills` (the Skills page) | **187** |
| `app.routers.mc_ui._summarize_skills()["skill_count"]` (the Mission Control tab) | **194** |
| `scripts.skill_lint.lint()["total"]` (the weekly report, task #70) | **194** |

The route was wrong in *both* directions at once: it listed the five
`status: archived` skills (`groundskeeper-loop`, `groundskeeper-research`,
`nightly-behavior-test`, `nightly-morning-briefing`, `nightly-prompt-audit`) and
dropped seven live ones (`alfie-monitoring`, `backlog-triage`,
`email-calendar-monitoring`, `github-watch`, `python-library-pipeline`,
`ralph-loop`, `workspace-audit`). The drop had nothing to do with quarantine: those
seven carry `metadata:\n  openclaw: null`, and
`meta.get("hermes", meta.get("openclaw", {}))` returns `None` for a key that is
present with a null value, so the next line's `hermes_meta.get("category")` raised
`AttributeError` and the route's bare `except Exception: continue` turned the row
into "this skill does not exist". A front-matter shape the author did not anticipate
deleted a row silently, which is the expensive half of the defect: every later
reader trusts the count.

The fix is one predicate in `agent_mcp.skills` — `iter_active_skills()` — and every
surface consuming it, so "dot-prefixed directory", "quarantined by `status:`" and
"this name is already taken by an earlier root" are decided in exactly one place.
The knobs below are deliberately the walker's own module globals: a test that moves
`agent_mcp.skills.SKILLS_DIRS` moves *all five* surfaces together, which is the
property this file exists to protect.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import agent_mcp.skills as skills_mod
import prompt_builder as pb
from app.routers.mc_ui import _summarize_skills
from app.routers.skills import get_skills

ROOT = Path(__file__).resolve().parents[1]


# ── the fixture: one tree, all five surfaces pointed at it ────────────────────

def _skill(dir_path: Path, *, status: str | None = None, description: str,
           extra_front_matter: str = "") -> Path:
    """Write a SKILL.md and return its path."""
    dir_path.mkdir(parents=True, exist_ok=True)
    lines = ["---", "name: placeholder", f"description: {description}"]
    if status:
        lines.append(f"status: {status}")
    lines.append(extra_front_matter.rstrip())
    lines += ["---", "", f"BODY of {dir_path.name}."]
    skill_file = dir_path / "SKILL.md"
    skill_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return skill_file


@pytest.fixture
def skill_tree(tmp_path, monkeypatch):
    """Two roots holding every shape the walker must have one opinion about.

    `alpha` and `beta` are ordinary live skills; `.archived/x` is the archive the
    tab used to count by accident; `retired` is pulled from circulation by its
    front matter; `shared` exists in both roots so first-root-wins is observable;
    `null-openclaw` is the front-matter shape that used to delete a row.
    """
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    _skill(root_a / "alpha", description="Use when alpha.")
    _skill(root_b / "beta", description="Use when beta.")
    _skill(root_a / ".archived" / "x", description="Archived by relocation.")
    _skill(root_a / "retired", description="Use when retired.", status="archived")
    _skill(root_a / "shared", description="Root A wins.")
    _skill(root_b / "shared", description="Root B loses.")
    _skill(root_b / "null-openclaw", description="Use when null.",
           extra_front_matter="metadata:\n  openclaw: null")
    monkeypatch.setattr(skills_mod, "SKILLS_DIRS", [root_a, root_b])
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)
    return tmp_path


def _walked() -> list[skills_mod.ActiveSkill]:
    return list(skills_mod.iter_active_skills())


async def _route_skills() -> list[dict]:
    response = await get_skills()
    return json.loads(response.body)["workspace"]


async def _route_names() -> set[str]:
    return {row["name"] for row in await _route_skills()}


def _index_names() -> set[str]:
    """The names the prompt advertises, parsed off the rendered index line."""
    index = pb._load_skills_index()
    assert index, "the prompt advertises no skills at all, so equality proves nothing"
    prefix = "Available skills: "
    assert index.startswith(prefix), index[:60]
    return set(index[len(prefix):].split(", "))


# ── clause 1: one walker, three behaviours ───────────────────────────────────

def test_the_walker_skips_dot_dirs_honours_quarantine_and_prefers_the_first_root(skill_tree):
    """`iter_active_skills` decides all three rules at once (clause 1).

    `.archived/x` is skipped because the directory is dot-prefixed, `retired`
    because its `status:` is in `_QUARANTINE_STATUSES`, and `shared` is yielded
    exactly once with root A's copy, which is the copy `skills_read` would serve.
    """
    walked = _walked()
    names = [s.name for s in walked]

    assert "x" not in names, "a skill under a dot-prefixed directory was walked"
    assert "retired" not in names, "a quarantined skill was walked"
    assert names.count("shared") == 1, f"first-root-wins did not dedupe: {names}"
    shared = next(s for s in walked if s.name == "shared")
    assert shared.skill_file.parent.parent == skill_tree / "root-a"
    assert shared.frontmatter["description"] == "Root A wins."
    assert sorted(names) == ["alpha", "beta", "null-openclaw", "shared"]
    assert all(isinstance(s.skill_file, Path) and s.skill_file.name == "SKILL.md"
               for s in walked)


def test_an_overlay_root_shadows_the_canonical_roots(tmp_path, monkeypatch):
    """The `overlay=` argument is autoresearch's whole mechanism: a variant skill
    of the same name replaces the canonical one for that build, not adds to it."""
    canonical = tmp_path / "canonical"
    _skill(canonical / "alpha", description="Canonical alpha.")
    monkeypatch.setattr(skills_mod, "SKILLS_DIRS", [canonical])

    overlay = tmp_path / "overlay" / "skills"
    _skill(overlay / "alpha", description="Variant alpha.")

    with_overlay = list(skills_mod.iter_active_skills(overlay=tmp_path / "overlay"))
    assert [s.name for s in with_overlay] == ["alpha"]
    assert with_overlay[0].frontmatter["description"] == "Variant alpha."
    assert list(skills_mod.iter_active_skills())[0].frontmatter["description"] == (
        "Canonical alpha.")


def test_prompt_builder_restates_neither_the_set_nor_the_rule(skill_tree):
    """One walker means one vocabulary (#1294).

    `prompt_builder` used to hold the quarantine set twice — an import of
    `agent_mcp.skills`' and a fallback literal behind a `try/except` — plus its own
    line-by-line reading of the `status:` key. All three were kept in step only by
    an equality assertion in one test, which is how a rule ends up differing between
    the prompt and the loader that serves it. The set now lives in exactly one
    module and the predicate is that module's function object.
    """
    assert not hasattr(pb, "_QUARANTINE_STATUSES"), (
        "prompt_builder carries a copy of the quarantine vocabulary again; the "
        "walker owns it")
    assert pb._is_quarantined_skill is skills_mod.is_quarantined_skill_file
    assert pb._is_quarantined_skill(
        skill_tree / "root-a" / "retired" / "SKILL.md") is True
    assert pb._is_quarantined_skill(
        skill_tree / "root-a" / "alpha" / "SKILL.md") is False


# ── clause 2: the route reports exactly the advertised set ───────────────────

async def test_the_route_reports_exactly_the_advertised_index(skill_tree):
    """One tree, two surfaces, one set (clause 2)."""
    route = await _route_names()
    assert route == _index_names()
    assert route == {"alpha", "beta", "null-openclaw", "shared"}
    assert "retired" not in route


async def test_the_live_surfaces_report_one_set():
    """The same equality over the real vault, which is what the acceptance names.

    Not marked `live_vault`: the marker is deselected by the gate
    (`scripts/automod/gate.py` runs `-m "not live_vault"`), and a pin that only a
    human happens to run is how the route stayed wrong twice. The coupling is paid
    for the same way `tests/test_archived_skill_artifacts.py` pays it — every
    assertion is structural, and each one says out loud when it measured nothing.
    """
    roots = [d for d in skills_mod.SKILLS_DIRS if d.is_dir()]
    assert roots, f"no skill root exists under {skills_mod.SKILLS_DIRS}"

    # A deliberately independent walk: the raw `iterdir()` the five surfaces used
    # to do, used here as the control the walker is diffed against. Quarantine is
    # still decided by the walker's own predicate, so this is one opinion about the
    # set and one about the location, not two opinions about what "live" means.
    on_disk: dict[str, Path] = {}
    for root in roots:
        for entry in sorted(root.iterdir()):
            skill_file = entry / "SKILL.md"
            if not entry.is_dir() or entry.name.startswith(".") or not skill_file.is_file():
                continue
            on_disk.setdefault(entry.name, skill_file)
    quarantined = {name for name, f in on_disk.items()
                   if skills_mod.is_quarantined_skill_file(f)}
    live = set(on_disk) - quarantined

    assert quarantined, (
        "no quarantined skill exists on disk, so 'the route excludes quarantined "
        "skills' cannot fail here — the five archived skills are the point of "
        "this assertion, and if the vault has none the assertion is vacuous")
    assert live, "no live skill exists on disk"

    route = await _route_names()
    assert route == live, (
        f"route-vs-disk: hidden={sorted(live - route)[:10]} "
        f"shown-that-should-not-be={sorted(route - live)[:10]}")
    assert _index_names() == route, "the prompt advertises a different set than the page lists"
    assert _summarize_skills()["skill_count"] == len(route)


# ── clause 3: an unexpected front-matter shape cannot delete a row ───────────

async def test_a_null_openclaw_block_does_not_delete_the_row(skill_tree):
    """`metadata: {openclaw: null}` with no top-level `category` still lists.

    The seven live skills the route dropped on 2026-09-21 all carry that shape.
    The expected outcome is a row with an empty category, not an absent row.
    """
    rows = {row["name"]: row for row in await _route_skills()}
    assert "null-openclaw" in rows, sorted(rows)
    assert rows["null-openclaw"]["category"] == ""
    assert rows["null-openclaw"]["description"] == "Use when null."
    assert rows["null-openclaw"]["location"].endswith("root-b/null-openclaw")


def test_no_live_skill_needs_the_hermes_fallback_the_route_dropped():
    """Why `metadata.hermes` / `metadata.openclaw` left the route rather than being
    null-guarded in place (#1294).

    Measured over the real roots before the change, every live skill carries its
    `description` and `category` at the top level, so the fallback supplied nothing —
    the one thing it did supply was `None` from a null-valued key, which is what ate
    seven rows. Removing it is only safe while that measurement holds, so it is
    re-run here instead of trusted.

    A skill that ever does move those fields under `hermes:` should be fixed, not
    catered to: `agent_mcp.skills._load_skill`, which decides what `skills_search`
    can retrieve, has only ever read the top-level fields — such a skill is already
    unreachable through retrieval, and the Skills page agreeing with that is the
    correct outcome. If this fails deliberately, restore the fallback in
    `app/routers/skills.py` and say why in its docstring.
    """
    would_have_been_used = []
    for active in skills_mod.iter_active_skills():
        fm = active.frontmatter
        metadata = fm.get("metadata")
        if not isinstance(metadata, dict):
            continue
        hermes_meta = metadata.get("hermes", metadata.get("openclaw", {}))
        if not isinstance(hermes_meta, dict):
            continue
        for field in ("description", "category"):
            if not fm.get(field) and hermes_meta.get(field):
                would_have_been_used.append(f"{active.name}.{field}")
    assert would_have_been_used == [], (
        "these live skills keep a description or category only under a "
        f"hermes/openclaw metadata block: {would_have_been_used}")


async def test_metadata_that_is_not_a_mapping_keeps_its_row(skill_tree, tmp_path):
    """The same failure one shape over: `metadata:` holding a string, and a
    front-matter block that is not parseable YAML at all."""
    odd = skill_tree / "root-b" / "broken-front-matter"
    odd.mkdir(parents=True)
    (odd / "SKILL.md").write_text(
        "---\nname: broken-front-matter\ndescription: Use when broken.\n"
        "metadata: [this, is, a, list]\n---\n\nBODY.\n", encoding="utf-8")
    unparseable = skill_tree / "root-b" / "unparseable-front-matter"
    unparseable.mkdir(parents=True)
    (unparseable / "SKILL.md").write_text(
        "---\nname: [unclosed\n  description: nonsense\n: ---\n\nBODY.\n",
        encoding="utf-8")

    rows = {row["name"]: row for row in await _route_skills()}
    assert {"broken-front-matter", "unparseable-front-matter"} <= set(rows), sorted(rows)
    assert rows["broken-front-matter"]["category"] == ""


# ── clause 4: the tab count is the route count, and the archive cannot leak ──

async def test_the_tab_count_is_the_route_count(skill_tree):
    """`_summarize_skills` used to count `entry.is_dir() and (entry/'SKILL.md').exists()`
    per top-level entry, which excluded `.archived` only because the archive
    happens to keep per-skill subdirectories (clause 4)."""
    route = await _route_names()
    assert _summarize_skills()["skill_count"] == len(route)
    assert _summarize_skills()["skill_count"] == 4


async def test_a_skill_file_directly_under_a_dot_dir_changes_nothing(skill_tree):
    """The accident, made impossible: a `SKILL.md` at the *top* of `.archived/`
    used to add a row to the tab count, because `.archived` itself satisfied
    "is a directory containing SKILL.md"."""
    before_route = await _route_names()
    before = _summarize_skills()["skill_count"]
    # The invariance below is only worth anything if the number being held still
    # is the right one: four live skills, not the six directories on disk.
    assert (before, len(before_route)) == (4, 4), (before, sorted(before_route))
    stray = skill_tree / "root-a" / ".archived" / "SKILL.md"
    stray.write_text("---\nname: stray\ndescription: Use when stray.\n---\n\nBODY.\n",
                     encoding="utf-8")

    assert _summarize_skills()["skill_count"] == before, "the archive leaked into the count"
    assert await _route_names() == before_route


# ── clause 5: skill_lint's total comes from the same walker ─────────────────

def _load_skill_lint():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_under_test", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_skill_lint_lints_the_walked_set_and_still_reports_defects(skill_tree, tmp_path):
    """Narrowing the linted set must not blind the lint (clause 5).

    Four live skills, one of them naming a tool the aggregator does not advertise;
    the archived skill carries the same phantom name, so a lint that still counts
    it is a lint that never saw the quarantine rule.
    """
    skill_lint = _load_skill_lint()
    for name in ("alpha", "retired"):
        body = (skill_tree / "root-a" / name / "SKILL.md").read_text(encoding="utf-8")
        (skill_tree / "root-a" / name / "SKILL.md").write_text(
            body + "\nThe tool `web_search` does the lookup.\n", encoding="utf-8")

    result = skill_lint.lint()
    assert result["total"] == 4, result["total"]

    phantom_names = {row["name"] for row in result["phantom"]}
    assert "alpha" in phantom_names, (
        "a live skill naming a tool that does not exist went unreported — the "
        "category exists precisely to catch this and the narrowed set must not "
        "cost it")
    assert "retired" not in phantom_names, "a quarantined skill was linted"

    dead = skill_tree / "root-b" / "no-front-matter"
    dead.mkdir(parents=True)
    (dead / "SKILL.md").write_text("# just a heading\n\nno front matter at all\n",
                                   encoding="utf-8")
    result = skill_lint.lint()
    assert result["total"] == 5
    assert {row["name"] for row in result["dead"]} == {"no-front-matter"}


def test_skill_lint_still_boots_when_invoked_as_the_nightly_invokes_it(tmp_path):
    """Task #70 runs `python …/scripts/skill_lint.py` by path, which puts `scripts/`
    on `sys.path` and *not* the repo root — so the walker import needs its own
    bootstrap, and the report path follows `$HOME`, which is redirected here."""
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "skill_lint.py")],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ModuleNotFoundError" not in proc.stderr
    assert "Totals:" in proc.stdout, proc.stdout[-1000:]
    assert (tmp_path / "obsidian" / "autonomy" / "skill-lint-report.md").exists()
