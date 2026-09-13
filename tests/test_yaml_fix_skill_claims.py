"""`skills/autonomy-pipeline-yaml-fix/SKILL.md` must describe the code that exists.

The skill is `status: active` and matched by `skills_search`, so an autonomy run
can follow it verbatim. Written on 2026-04-03, it had drifted three ways (backlog
#488, confirmed at triage 2026-09-12):

* its "Scripts Requiring This Fix" list named a script deleted in vault-side
  commit `0b3f00b` (2026-09-03, "kg phase 7: delete what nothing runs") and
  missed one of the two scripts that still carry the fallback class;
* its Testing block invoked the system interpreter instead of the venv the
  pipeline is supported on;
* its rationale asserted that the yaml module is unavailable in this
  environment. Measured 2026-09-12: PyYAML 6.0.3 on Python 3.14.7
  (`/usr/bin/python3`) and on Python 3.12.14 (`.venvs/lloyd/bin/python`). The
  trigger for the fallback no longer reproduces on either interpreter, so the
  claim was what was holding a partial YAML parser in the tree. Whether the
  class goes outright is #484's call, not this file's.

Only claims that mislead an operator are pinned — not prose. The checks read the
live vault, hence `live_vault`: an hourly autoresearch promotion or a nightly
skills pass can rewrite the file between rounds (see `pytest.ini`). If #484
removes the fallback class, the carrier set goes empty and the skill becomes a
historical note; that item owns updating this file along with it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SKILLS_DIR = Path.home() / "obsidian" / "skills"
YAML_SKILL = SKILLS_DIR / "autonomy-pipeline-yaml-fix" / "SKILL.md"
PIPELINE_SKILL = SKILLS_DIR / "autonomy-data-pipeline" / "SKILL.md"
# The venv lives in the live checkout, not in a round's worktree (it is
# gitignored), so it is resolved from $HOME like the skill prose does.
VENV_PY = Path.home() / "lloyd" / ".venvs" / "lloyd" / "bin" / "python"

# The literal strings the acceptance grep looks for across every skill.
ABSENCE_CLAIMS = (
    "No module named 'yaml'",
    "not available in the restricted",
    "lacks deps",
)
# A `~/lloyd/...py` path as it is spelled in skill prose.
NAMED_PATH = re.compile(r"~/lloyd/[A-Za-z0-9_./-]+\.py")

pytestmark = [
    pytest.mark.live_vault,
    pytest.mark.skipif(not YAML_SKILL.exists(), reason="vault skill not present"),
]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _section(text: str, heading: str) -> str:
    """Markdown body under `heading`, up to the next heading or rule."""
    start = text.index(heading) + len(heading)
    rest = text[start:]
    cut = re.search(r"^#{1,3} |^---\s*$", rest, re.M)
    return rest[: cut.start()] if cut else rest


def _fenced(text: str) -> list[str]:
    """Contents of every ``` fence in `text` (odd/even pairing)."""
    parts = text.split("```")
    return [parts[i] for i in range(1, len(parts), 2)]


def _fallback_carriers() -> set[str]:
    """Scripts whose `try: import yaml / except ImportError: class yaml` block
    exists on disk right now, as skill-prose paths."""
    carriers = set()
    for py in (ROOT / "scripts").rglob("*.py"):
        try:
            body = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "class yaml:" in body and "except ImportError" in body:
            carriers.add("~/lloyd/" + py.relative_to(ROOT).as_posix())
    return carriers


# ── code truth the prose is pinned to ───────────────────────────────────────


def test_exactly_two_scripts_still_carry_the_fallback_class():
    """Disk truth behind clause 2: two carriers, both under `scripts/`."""
    assert _fallback_carriers() == {
        "~/lloyd/scripts/memory/rebuild_index.py",
        "~/lloyd/scripts/memory/next-gen-memory/relations_index.py",
    }


# ── clause 1: the deleted script is gone from the skills ─────────────────────


def test_only_the_historical_mention_of_the_deleted_script_survives():
    """`semantic_relationships` remains only as prose in autonomy-data-pipeline
    saying the old Step 2 ran it and it is gone."""
    assert SKILLS_DIR.exists(), f"no skills dir at {SKILLS_DIR}"
    hits = sorted(
        md.relative_to(SKILLS_DIR).as_posix()
        for md in SKILLS_DIR.rglob("*.md")
        if "semantic_relationships" in _text(md)
    )
    assert hits == ["autonomy-data-pipeline/SKILL.md"], hits
    line = next(
        ln for ln in _text(PIPELINE_SKILL).splitlines() if "semantic_relationships" in ln
    )
    assert line.lstrip().startswith(">") and "old Step 2" in line, line


# ── clause 2: the list names the real carriers, and nothing else ─────────────


def test_the_scripts_section_names_the_real_carriers_only():
    named = set(NAMED_PATH.findall(_section(_text(YAML_SKILL), "## Scripts Requiring This Fix")))
    assert named == _fallback_carriers(), named ^ _fallback_carriers()
    assert "Apply the same patch to all three scripts" not in _text(YAML_SKILL)


# ── clause 3: every path it names resolves ──────────────────────────────────


def test_every_script_path_the_skill_names_resolves_on_disk_and_at_head():
    """A skill that orders a patch to a deleted file is the defect this is."""
    named = sorted(set(NAMED_PATH.findall(_text(YAML_SKILL))))
    assert named, "the skill names no script at all — did the section move?"
    for prose_path in named:
        rel = prose_path.removeprefix("~/lloyd/")
        assert (Path.home() / "lloyd" / rel).is_file(), (
            f"{prose_path} named in the skill, but absent from disk"
        )
        probe = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{rel}"],
            capture_output=True, text=True,
        )
        assert probe.returncode == 0, f"{prose_path} is not at HEAD: {probe.stderr.strip()}"



# ── clause 4: the Testing block runs the venv, not the system interpreter ────


def test_the_testing_block_invokes_the_venv_interpreter():
    """`python3` may appear in prose; inside a fence it is the wrong
    interpreter, which is what the acceptance check pins."""
    section = _section(_text(YAML_SKILL), "## Testing")
    blocks = _fenced(section)
    assert blocks, "the Testing section has no fenced command"
    assert str(VENV_PY) in "\n".join(blocks), "the venv interpreter is not in the command"
    for block in _fenced(_text(YAML_SKILL)):
        assert "python3" not in block, block


def test_the_command_the_skill_publishes_actually_runs():
    """Across the interpreter seam: the script the Testing block tells a reader
    to run, run under the interpreter it names, exits 0."""
    script = "scripts/memory/next-gen-memory/relations_index.py"
    assert script in _text(YAML_SKILL), "the Testing block no longer names a runnable script"
    proc = subprocess.run(
        [str(VENV_PY), script, "--help"], cwd=ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: relations_index.py" in proc.stdout


# ── clause 5: the preferred fix, and whose call the removal is ───────────────


def test_the_skill_says_the_venv_is_the_fix_and_defers_the_class_to_484():
    text = _text(YAML_SKILL)
    preferred = [
        ln for ln in text.splitlines()
        if "preferred" in ln.lower() and ".venvs/lloyd/bin/python" in ln
    ]
    assert preferred, "no line states that running under the venv is the preferred fix"
    assert re.search(r"#?484", text), "the skill does not point at #484 for the removal decision"


# ── clause 6: the yaml-absence rationale is retired everywhere ────────────────


def test_no_skill_still_claims_the_yaml_module_is_missing():
    offenders = []
    for md in sorted(SKILLS_DIR.rglob("*.md")):
        for n, line in enumerate(_text(md).splitlines(), 1):
            if any(claim in line for claim in ABSENCE_CLAIMS):
                offenders.append(f"{md.relative_to(SKILLS_DIR)}:{n}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)


def test_pyyaml_imports_under_the_supported_interpreter():
    """The measured fact the retired rationale contradicted: PyYAML 6.0.3 on
    Python 3.12.14 in the venv, so the fallback class never fires there."""
    proc = subprocess.run(
        [str(VENV_PY), "-c", "import sys, yaml; print(sys.version_info[0], yaml.__version__)"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    python_major, _, version = proc.stdout.strip().partition(" ")
    assert python_major == "3"
    assert tuple(int(n) for n in version.split(".")[:2]) >= (6, 0), version
