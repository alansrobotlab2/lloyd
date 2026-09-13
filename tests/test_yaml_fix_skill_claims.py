"""`skills/autonomy-pipeline-yaml-fix/SKILL.md` must describe the code that exists.

The skill is `status: active` and matched by `skills_search`, so an autonomy run
can follow it verbatim. Written on 2026-04-03, it had drifted three ways (backlog
#488, confirmed at triage 2026-09-12):

* its "Scripts Requiring This Fix" list named a script deleted in vault commit
  `0b3f00b` (2026-09-03, "kg phase 7: delete what nothing runs") and missed one
  of the two scripts that still carry the fallback class;
* its Testing block invoked the system interpreter instead of the venv the
  pipeline is supported on;
* its rationale asserted that the yaml module is unavailable in this
  environment. Measured 2026-09-12: PyYAML 6.0.3 imports on Python 3.14.7
  (`/usr/bin/python3`) and on Python 3.12.14 (`.venvs/lloyd/bin/python`), so the
  fallback's trigger no longer reproduces on either interpreter, and the claim
  was what was keeping a partial YAML parser in the tree.

Whether the fallback classes go outright is backlog #484's call, not this
file's; the skill now defers to it, and one test below pins that it defers.

Scope of the scans: the acceptance greps are `grep -rn` over every file under
`~/obsidian/skills/`, so the two whole-directory scans here walk every file,
not just `*.md` — an absence claim parked in a `.py` or `.sh` would otherwise
pass the test while failing the clause. Measured 2026-09-13, the only file under
that tree still mentioning the deleted script is `autonomy-data-pipeline/SKILL.md`,
and no file under it still claims yaml is missing.

These assertions read the live vault, so they can go red from a nightly skills
pass rather than from the change under review. That is the same trade
`tests/test_skill_tool_names.py` already makes for live skill prose, and it is
deliberate: the gate runner hardcodes `-m "not live_vault"`, so a marked check
is deselected from the run meant to enforce it and pins nothing. Nothing here
skips either — with no vault at all the reads fail with `NO_VAULT` in the
message, which is a real answer rather than a green-looking absence of evidence.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SKILLS_DIR = Path.home() / "obsidian" / "skills"
YAML_SKILL = SKILLS_DIR / "autonomy-pipeline-yaml-fix" / "SKILL.md"
PIPELINE_SKILL = SKILLS_DIR / "autonomy-data-pipeline" / "SKILL.md"
# The venv lives in the live checkout, not in a round's worktree (it is
# gitignored), so it is resolved from $HOME the way the skill prose spells it.
VENV_PY = Path.home() / "lloyd" / ".venvs" / "lloyd" / "bin" / "python"

# The literal strings the acceptance grep looks for across the skills tree.
ABSENCE_CLAIMS = (
    "No module named 'yaml'",
    "not available in the restricted",
    "lacks deps",
)
# A `~/lloyd/...py` path as it is spelled in skill prose.
NAMED_PATH = re.compile(r"~/lloyd/[A-Za-z0-9_./-]+\.py")

# Trees under ROOT that are not source: vendored deps, caches, build output,
# git internals, and the venv (which is gitignored and lives inside the
# checkout, so a raw walk would descend tens of thousands of site-packages
# files looking for one class definition).
_NOT_SOURCE = {
    ".git", ".venv", ".venvs", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", "dist", "build",
}


NO_VAULT = f"no skills tree at {SKILLS_DIR}: these assertions exist to police that tree"


def _text(path: Path) -> str:
    assert path.is_file(), NO_VAULT
    return path.read_text(encoding="utf-8", errors="replace")


def _skill_files() -> list[Path]:
    """Every file under the skills tree, as `grep -rn` would walk it."""
    files = []
    for path in sorted(SKILLS_DIR.rglob("*")):
        if path.is_file() and path.stat().st_size <= 2_000_000:
            files.append(path)
    return files


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


# The block as the two carriers actually write it. Matching the shape rather
# than the substrings `"class yaml:"` and `"except ImportError"` is what keeps
# this file — a test about that block, containing both substrings — out of its
# own result set.
_FALLBACK_BLOCK = re.compile(
    r"^\s*try:\s*\n\s*import\s+yaml\s*\n\s*except\s+ImportError:.*?^\s*class\s+yaml\s*:",
    re.M | re.S,
)


def _fallback_carriers() -> set[str]:
    """Scripts under ROOT whose `try: import yaml / except ImportError: class
    yaml` block exists on disk right now, as skill-prose paths."""
    carriers: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _NOT_SOURCE]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            py = Path(dirpath) / name
            try:
                body = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _FALLBACK_BLOCK.search(body):
                carriers.add("~/lloyd/" + py.relative_to(ROOT).as_posix())
    return carriers


# ── code truth the prose is pinned to ───────────────────────────────────────


def test_exactly_two_scripts_still_carry_the_fallback_class():
    """Disk truth behind clause 2, over the whole checkout: two carriers, both
    under `scripts/`. The skill's claim that they are the only two is only as
    good as this detector, which is why it walks the repo and not one subtree."""
    assert _fallback_carriers() == {
        "~/lloyd/scripts/memory/rebuild_index.py",
        "~/lloyd/scripts/memory/next-gen-memory/relations_index.py",
    }


# ── clause 1: the deleted script is gone from the skills ─────────────────────


def test_only_the_historical_mention_of_the_deleted_script_survives():
    """`semantic_relationships` survives only as prose in autonomy-data-pipeline
    saying the old Step 2 ran it and it is gone."""
    hits = sorted(
        f.relative_to(SKILLS_DIR).as_posix()
        for f in _skill_files()
        if "semantic_relationships" in _text(f)
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


def _bullets(section: str) -> list[str]:
    """The section's `- ` items, with their continuation lines joined, so a
    bullet whose path and line-range sit on different wrapped lines is still
    one unit."""
    out: list[str] = []
    for line in section.splitlines():
        if line.lstrip().startswith("- "):
            out.append(line.lstrip()[2:].strip())
        elif out and line.strip():
            out[-1] += " " + line.strip()
    return out


def test_the_line_ranges_the_skill_cites_are_where_the_block_actually_is():
    """The two bullets cite the block by line number — `at lines 17-21` and
    `at lines 18-22`. The names being right is clause 2; a stale *range* is the
    same defect one step down, so each cited range is checked against the file
    rather than left accurate-by-luck: the first number is the `try:` line that
    opens the block, the last is the `class yaml:` line that closes its header.
    A round that shifts either script by one line fails here and is the round
    that has to move the number."""
    checked = 0
    for bullet in _bullets(_section(_text(YAML_SKILL), "## Scripts Requiring This Fix")):
        cited = re.search(r"\blines? (\d+)-(\d+)\b", bullet)
        if not cited:
            continue
        paths = NAMED_PATH.findall(bullet)
        assert paths, f"bullet cites a line range but names no script: {bullet!r}"
        rel = paths[0].removeprefix("~/lloyd/")
        body = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        block = _FALLBACK_BLOCK.search(body)
        assert block, f"{rel} no longer carries the block the skill cites"
        opener = body.count("\n", 0, block.start()) + 1
        closer = body.count("\n", 0, body.index("class yaml", block.start())) + 1
        assert (opener, closer) == (int(cited.group(1)), int(cited.group(2))), (
            f"{rel}: the skill cites lines {cited.group(1)}-{cited.group(2)}, "
            f"the block is at {opener}-{closer}"
        )
        checked += 1
    assert checked == 2, f"expected both carriers' bullets to cite a range, got {checked}"


# ── clause 3: every path it names resolves ──────────────────────────────────


def test_every_script_path_the_skill_names_resolves_on_disk_and_at_head():
    """A skill that orders a patch to a deleted file is the defect this is. The
    live checkout is consulted for both halves: the file on disk, and the same
    path at HEAD, which is what prose is allowed to cite."""
    live = Path.home() / "lloyd"
    named = sorted(set(NAMED_PATH.findall(_text(YAML_SKILL))))
    assert named, "the skill names no script at all — did the section move?"
    for prose_path in named:
        rel = prose_path.removeprefix("~/lloyd/")
        assert (live / rel).is_file(), f"{prose_path} named in the skill, but absent from disk"
        probe = subprocess.run(
            ["git", "-C", str(live), "cat-file", "-e", f"HEAD:{rel}"],
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


def _test_command_scripts() -> list[str]:
    """Script paths named inside a Testing fence block — not anywhere in the
    file, or a stale bullet in another section would vouch for the command."""
    found = []
    for block in _fenced(_section(_text(YAML_SKILL), "## Testing")):
        for token in re.findall(r"[A-Za-z0-9_./-]+\.py", block):
            found.append(token)
    return found


def test_the_command_the_skill_publishes_actually_runs():
    """Across the interpreter seam: each script the Testing block names, run
    under the interpreter that block names, exits 0 and prints usage. The
    block is parsed, so a command that quietly changed script while the prose
    bullet stayed the same cannot pass."""
    scripts = _test_command_scripts()
    assert scripts, "the Testing section's fenced command names no .py file"
    for rel in scripts:
        script = ROOT / rel
        assert script.is_file(), f"Testing block names {rel}, absent from the checkout"
        proc = subprocess.run(
            [str(VENV_PY), rel, "--help"], cwd=ROOT, capture_output=True, text=True
        )
        assert proc.returncode == 0, f"{rel} --help: {proc.stderr}"
        assert "usage:" in proc.stdout, f"{rel} --help printed no usage: {proc.stdout[:200]}"


# ── clause 5: the preferred fix, and whose call the removal is ───────────────


def test_the_skill_says_the_venv_is_the_fix_and_defers_the_class_to_484():
    text = _text(YAML_SKILL)
    preferred = [
        ln for ln in text.splitlines()
        if "preferred" in ln.lower() and ".venvs/lloyd/bin/python" in ln
    ]
    assert preferred, "no line states that running under the venv is the preferred fix"
    # The other half of clause 5: the preference has to exclude the alternative,
    # or "preferred" is a suggestion an agent can take or leave.
    against = [
        ln for ln in text.splitlines()
        if re.search(r"\bdo not (add|keep)\b", ln.lower()) and "pars" in ln.lower()
    ]
    assert against, "no line tells the reader not to add or keep a partial YAML parser"
    # Backlog reference, not any bare "484": the hash is required, and the
    # deferral has to be the decision about deleting the class.
    flat = " ".join(text.split())
    assert re.search(r"#[*]{0,2}484\b[^.]{0,160}?\b(decision|call)\b", flat), (
        "the skill does not defer deleting the fallback class to backlog #484"
    )


# ── clause 6: the yaml-absence rationale is retired everywhere ────────────────


def test_no_skill_still_claims_the_yaml_module_is_missing():
    offenders = []
    for path in _skill_files():
        for n, line in enumerate(_text(path).splitlines(), 1):
            if any(claim in line for claim in ABSENCE_CLAIMS):
                offenders.append(f"{path.relative_to(SKILLS_DIR)}:{n}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)


def test_pyyaml_imports_under_the_supported_interpreter():
    """The measured fact the retired rationale contradicted: under the
    interpreter the skill names, `import yaml` succeeds and reports the same
    version the skill's Testing block tells the reader to expect. Both halves
    can fail — an environment that loses PyYAML reddens the import, and a
    version bump reddens the prose until the page is updated with it."""
    probe = "import sys, yaml; print(sys.executable); print(yaml.__version__)"
    proc = subprocess.run(
        [str(VENV_PY), "-c", probe], capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        f"`import yaml` failed under the interpreter the skill names: {proc.stderr}"
    )
    executable, _, version = proc.stdout.strip().partition("\n")
    assert Path(executable) == VENV_PY, f"not the venv interpreter: {executable}"
    stated = re.findall(r"PyYAML (\d+\.\d+\.\d+)", _text(YAML_SKILL))
    assert stated, "the skill states no PyYAML version to expect"
    assert set(stated) == {version}, (
        f"interpreter reports PyYAML {version}, the skill tells the reader to expect {stated}"
    )
