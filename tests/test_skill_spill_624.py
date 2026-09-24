"""The five sampled skills are spilled into sibling files, and nothing was lost (#624).

"Your skill is really a folder": the SKILL.md body is the index and the rules, the
detail lives beside it and is read when its index line says so. A spill can go
wrong three ways that nobody notices until a nightly run misbehaves, and each has
a test here:

* a section is dropped rather than moved — every `##`/`###` heading the pre-spill
  body carried must still be in the body or verbatim in a sibling;
* a sibling is written but never pointed at — each one needs an index line that
  names it and says when to read it, inside the chat injector's cut
  (`prefetch.SKILL_BODY_MAX`), or the capped route never learns it exists;
* a guardrail is moved out of the body — an upper-case hard-constraint line
  (`prefetch._HARD_CONSTRAINT_RE`, the carry-forward rule) must still be in the
  body, with its text intact (hard wraps may be undone).

The pre-image is the vault at `skill_lint.SPILL_BASELINE_BEFORE`, resolved from
git the way `spill_delta` resolves it — not `HEAD~`, which stops being the
pre-spill state the moment anything else is committed to the vault.

The vault is live and shared, so the corpus tests carry `live_vault` and the gate
skips them; `skill_folder_text`, which makes the lint read the folder, is pinned
hermetically.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_spill", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load()

VAULT = Path.home() / "obsidian"
_HEADING = re.compile(r"^#{2,3}\s+\S")


def _unfenced(text: str):
    fence = False
    for line in text.splitlines():
        if line.strip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if not fence:
            yield line


def _norm(text: str) -> str:
    return " ".join(text.split())


def _baseline_rev() -> str:
    out = subprocess.run(["git", "-C", str(VAULT), "rev-list", "-1",
                          f"--before={sl.SPILL_BASELINE_BEFORE}", "HEAD"],
                         capture_output=True, text=True, timeout=30)
    rev = out.stdout.strip()
    if out.returncode != 0 or not rev:
        pytest.skip(f"no vault commit before {sl.SPILL_BASELINE_BEFORE}")
    return rev


def _before(name: str) -> str:
    out = subprocess.run(["git", "-C", str(VAULT), "show",
                          f"{_baseline_rev()}:skills/{name}/SKILL.md"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        pytest.skip(f"skills/{name}/SKILL.md absent at the baseline")
    return out.stdout


def _now(name: str) -> tuple[str, list[Path]]:
    d = VAULT / "skills" / name
    if not (d / "SKILL.md").is_file():
        pytest.skip(f"{d} not present on this machine")
    siblings = sorted(p for p in d.glob("*.md") if p.name != "SKILL.md")
    return (d / "SKILL.md").read_text(encoding="utf-8"), siblings


def _body(text: str) -> str:
    return sl.parse_frontmatter(text)[1].strip("\n")


@pytest.fixture(scope="module", autouse=False)
def vault_git():
    if not (VAULT / ".git").exists():
        pytest.skip("the vault is not a git checkout here")


# ── the corpus: the five sampled skills as they are on disk ──────────────────

@pytest.mark.live_vault
@pytest.mark.parametrize("name", sl.SPILL_SAMPLE)
def test_the_body_is_within_the_ceiling(name, vault_git):
    text, _ = _now(name)
    lines = len(_body(text).splitlines())
    assert lines <= sl.MAX_BODY_LINES, f"{name}: {lines} body lines"


@pytest.mark.live_vault
@pytest.mark.parametrize("name", sl.SPILL_SAMPLE)
def test_every_removed_heading_is_verbatim_in_a_sibling(name, vault_git):
    before = _before(name)
    text, siblings = _now(name)
    body_heads = {ln.strip() for ln in _unfenced(_body(text)) if _HEADING.match(ln)}
    sibling_lines = {ln.strip() for p in siblings
                     for ln in p.read_text(encoding="utf-8").splitlines()}
    removed = [ln.strip() for ln in _unfenced(_body(before))
               if _HEADING.match(ln) and ln.strip() not in body_heads]
    lost = [h for h in removed if h not in sibling_lines]
    assert not lost, f"{name}: headings in neither the body nor a sibling: {lost}"


@pytest.mark.live_vault
@pytest.mark.parametrize("name", sl.SPILL_SAMPLE)
def test_each_sibling_has_an_index_line_inside_the_chat_cut(name, vault_git):
    text, siblings = _now(name)
    assert siblings, f"{name} has no sibling files — the spill has not been done"
    head = text[:sl.CHAT_SKILL_CUT]
    for sibling in siblings:
        lines = [ln for ln in head.splitlines() if sibling.name in ln]
        assert lines, (f"{name}: no line inside the first {sl.CHAT_SKILL_CUT} chars "
                       f"names {sibling.name}")
        # The condition: "read … when/before/after/only …" on the same line.
        assert any(re.search(r"\b[Rr]ead\b.*\b(when|before|after|only|at)\b", ln)
                   for ln in lines), (
            f"{name}: {sibling.name} is named but no line says when to read it: {lines}")


@pytest.mark.live_vault
@pytest.mark.parametrize("name", sl.SPILL_SAMPLE)
def test_hard_constraints_stay_in_the_body(name, vault_git):
    import prefetch
    before = _before(name)
    text, _ = _now(name)
    body = _norm(_body(text))
    hard = [ln.strip() for ln in _unfenced(_body(before))
            if prefetch._HARD_CONSTRAINT_RE.search(ln)]
    moved = [ln for ln in hard if _norm(ln) not in body]
    assert not moved, f"{name}: hard-constraint lines moved out of the body: {moved}"


@pytest.mark.live_vault
def test_the_spill_delta_is_nonzero_and_a_saving(vault_git):
    delta = sl.spill_delta(VAULT)
    rows = {r["name"]: r for r in delta["skills"]}
    assert set(rows) == set(sl.SPILL_SAMPLE)
    for name, row in rows.items():
        assert row["delta_chars"] is not None and row["delta_chars"] < 0, (name, row)
        assert row["siblings"], name
    assert delta["total_delta_chars"] < 0


# ── the lint reads the folder, so a spill hides nothing from it ──────────────

def test_skill_folder_text_appends_every_sibling_markdown(tmp_path):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "SKILL.md").write_text("body\n", encoding="utf-8")
    (d / "b.md").write_text("second\n", encoding="utf-8")
    (d / "a.md").write_text("first\n", encoding="utf-8")
    (d / "notes.txt").write_text("not markdown\n", encoding="utf-8")
    text = sl.skill_folder_text(d, "body\n")
    assert text.index("body") < text.index("first") < text.index("second")
    assert "not markdown" not in text
    assert text.count("body") == 1, "SKILL.md must not be read twice"


def test_lint_finds_a_phantom_tool_that_moved_into_a_sibling(tmp_path):
    """The counterfactual for reading the folder: the same phantom name, moved
    out of SKILL.md into a sibling, is still a finding."""
    from agent_mcp.skills import ActiveSkill
    d = tmp_path / "skills" / "spilled"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\ndescription: Use when testing a spill.\ntags: [t]\n---\n\n"
        "Read `detail.md` when you need the detail.\n", encoding="utf-8")
    (d / "detail.md").write_text("Call `web_search` for every lookup.\n", encoding="utf-8")
    rec = ActiveSkill(name="spilled", directory=d, frontmatter={})
    result = sl.lint(skill_records=[rec])
    assert [p["name"] for p in result["phantom"]] == ["spilled"], result["phantom"]
