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

The prose is not the artifact an agent reads. `agent_mcp/skills.py` parses the
front matter (`_parse_frontmatter`, `_load_skill`) and `skills_search` serves
that parsed description, while the prefetch injects `skill["body"]` — so a page
whose front matter degrades is not the page this file's text assertions read.
The seam section therefore asserts against what the loader returns and against
the `skills_search` result for the query an agent makes when a pipeline step
dies on yaml. That section found a live defect on landing: an unquoted ` #`
inside a plain YAML scalar opens a comment, so the description the loader served
stopped at "… deleting the classes is backlog" and the #484 deferral was
invisible in search results. Fixed in the page's front matter ("backlog item
484"), and pinned tree-wide by
`test_no_skill_description_loses_its_tail_to_a_yaml_comment`.

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

from app.paths import LIVE_CHECKOUT  # noqa: E402

SKILLS_DIR = Path.home() / "obsidian" / "skills"
YAML_SKILL = SKILLS_DIR / "autonomy-pipeline-yaml-fix" / "SKILL.md"
PIPELINE_SKILL = SKILLS_DIR / "autonomy-data-pipeline" / "SKILL.md"
# The venv lives in the live checkout, not in a round's worktree (it is
# gitignored). Resolved off the account home, not $HOME: a gate's
# `HOME=<round>/home` makes `~/lloyd` the worktree, which has no venv.
VENV_PY = LIVE_CHECKOUT / ".venvs" / "lloyd" / "bin" / "python"

# The literal strings the acceptance grep looks for across the skills tree.
ABSENCE_CLAIMS = (
    "No module named 'yaml'",
    "not available in the restricted",
    "lacks deps",
)
# A `~/lloyd/...py` path as it is spelled in skill prose.
NAMED_PATH = re.compile(r"~/lloyd/[A-Za-z0-9_./-]+\.py")
# An interpreter as it is spelled on a shell command line: any number of
# `/segment/` hops (or a `~/` / `./` root) in front of `python`, with optional
# version digits. Matches `/home/u/lloyd/.venvs/lloyd/bin/python` and `python3`,
# and never a script path like `relations_index.py`.
INTERPRETER = re.compile(r"(?:[~.]?/)?(?:[A-Za-z0-9_.-]+/)*python[0-9.]*")

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


# The walk is only ever bounded by a size cap, and the cap is never allowed to
# hide a file: see `_skill_files`.
_SIZE_CAP = 2_000_000


def _skill_files() -> list[Path]:
    """Every file under the skills tree, as `grep -rn` would walk it.

    Nothing is dropped quietly. A file over `_SIZE_CAP` would make the scan a
    different denominator than the acceptance grep, so rather than skip it (the
    shape that silently passes a clause it never read) this asserts: the scan
    fails, names the file, and says what to do. Largest file in the tree
    measured 2026-09-13 was 242 KB, so the assert is a guard against a future
    artifact, not a live constraint.
    """
    files = sorted(p for p in SKILLS_DIR.rglob("*") if p.is_file())
    # An empty tree would make every absence claim below vacuously true, so the
    # denominator is asserted before the contents are.
    assert files, f"{NO_VAULT} (walked {SKILLS_DIR} and found no files)"
    oversized = [p for p in files if p.stat().st_size > _SIZE_CAP]
    assert not oversized, (
        f"{[p.stat().st_size for p in oversized]} byte(s) exceed the {_SIZE_CAP} "
        f"read cap: {[p.relative_to(SKILLS_DIR).as_posix() for p in oversized]} — "
        "raise the cap, do not let the scan walk past them"
    )
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


def _testing_commands() -> list[tuple[str, str]]:
    """`(interpreter, script)` for each script named in a Testing fence block.

    Both halves are read out of the fence, not from a constant: an interpreter
    token (a `…/bin/python…` path or a bare `python`/`python3`) and the `.py`
    path on the same command line. Parsing the interpreter is what lets the test
    below honestly say "run under the interpreter the block names" — hardcoding
    the venv here would keep passing if the published command silently reverted
    to `python3`, which is the defect clause 4 exists to catch. Script paths
    come from the fence too, so a command that quietly changed script while the
    prose bullet stayed the same cannot pass.
    """
    pairs: list[tuple[str, str]] = []
    for block in _fenced(_section(_text(YAML_SKILL), "## Testing")):
        # A shell continuation is one command; tokenizing physical lines would
        # put the interpreter and the script on different lines and find a
        # command in neither.
        for line in re.sub(r"\\\s*\n\s*", " ", block).splitlines():
            tokens = [t for t in re.split(r"\s+", line.strip()) if t]
            interp = next((t for t in tokens if INTERPRETER.fullmatch(t)), None)
            for token in tokens:
                if token.endswith(".py") and interp is not None:
                    pairs.append((interp, token))
    return pairs


def test_the_command_the_skill_publishes_actually_runs():
    """Across the interpreter seam: each script the Testing fence names, run
    under the interpreter that same fence names, exits 0 and prints usage. The
    fence is the only source of both, so the published command is the thing
    being executed."""
    commands = _testing_commands()
    assert commands, "the Testing section's fenced command names no .py file"
    for interp_token, rel in commands:
        interpreter = (Path.home() / interp_token.removeprefix("~/")) if interp_token.startswith("~/") else Path(interp_token)
        assert interpreter.is_file(), f"the Testing fence names {interp_token}, which is not an interpreter on disk"
        script = ROOT / rel
        assert script.is_file(), f"Testing block names {rel}, absent from the checkout"
        proc = subprocess.run(
            [str(interpreter), rel, "--help"], cwd=ROOT, capture_output=True, text=True
        )
        assert proc.returncode == 0, f"{interpreter} {rel} --help: {proc.stderr}"
        assert "usage:" in proc.stdout, f"{interpreter} {rel} --help printed no usage: {proc.stdout[:200]}"


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


# ── seam: what the MCP process actually hands an autonomy run ────────────────


def test_the_skills_loader_serves_the_rewritten_page_not_a_degraded_copy():
    """The process boundary this change crosses. Nothing above imports
    `agent_mcp.skills`, and an autonomy run never reads the file: it gets
    `_load_skill`'s dict (the `skills_search` description and the body the
    prefetch injects) out of the MCP/prefetch processes.

    Two ways that degrades silently while every file-level assertion above
    still passes, both from `skills.py`:

    * `_parse_frontmatter` catches every exception and returns an empty dict
      (`agent_mcp/skills.py:51-54`), so one mis-indented continuation line in
      the rewritten `description:` yields a page that still circulates with an
      empty description and no `status` — the guidance nobody can find by
      searching for it;
    * a `status:` that lands in `_QUARANTINE_STATUSES`
      (`agent_mcp/skills.py:38,68-71`) drops the page from retrieval entirely,
      which for a page whose whole job is to stop an agent patching a dead
      script is the failure mode worth policing.

    So this runs on what the loader returns, not on the file: the description
    carries the venv guidance, the body it injects has no stale list, and
    `skills_search` — the tool call an autonomy run makes when a pipeline step
    dies on yaml — actually reaches the page.
    """
    import json

    from agent_mcp import skills

    loaded = skills._load_skill(SKILLS_DIR / "autonomy-pipeline-yaml-fix")
    assert loaded is not None, (
        "the loader quarantined the page, so no run can ever be shown the corrected "
        f"guidance; status: {_parse_status(_text(YAML_SKILL))!r}"
    )
    assert loaded["description"].strip(), (
        "front matter parsed to nothing — the page circulates with an empty description "
        "and cannot be found by skills_search"
    )
    assert "venv" in loaded["description"].lower(), loaded["description"]
    assert "484" in loaded["description"], (
        "the deferral is not in the string search returns. Note this is the seam "
        "that caught a live defect: an unquoted ` #` in a plain YAML scalar opens a "
        "comment, so '… is backlog #484' parsed to '… is backlog' and the pointer "
        "was invisible to every agent that found the page"
    )
    for claim in ABSENCE_CLAIMS:
        assert claim not in loaded["description"], f"description still asserts {claim!r}"
    # The body is what the prefetch injects as the skill hint.
    assert "Apply the same patch to all three scripts" not in loaded["body"]
    assert "semantic_relationships" not in loaded["body"], loaded["body"][:200]
    for block in _fenced(loaded["body"]):
        assert "python3" not in block, block
    # Cross the tool seam itself: the search an agent would run on the symptom.
    hits = json.loads(skills._skills_search(
        {"query": "yaml frontmatter fallback pipeline script", "max_results": 10}
    ))["results"]
    named = [h["name"] for h in hits]
    assert "autonomy-pipeline-yaml-fix" in named, f"skills_search never surfaced it: {named}"
    served = next(h for h in hits if h["name"] == "autonomy-pipeline-yaml-fix")
    assert served["description"] == loaded["description"], "the tool served a different description than the loader parsed"
    for claim in ABSENCE_CLAIMS:
        assert not any(claim in h["description"] for h in hits), (
            f"a skill surfaced on a yaml-pipeline query still asserts {claim!r}"
        )


def _parse_status(text: str) -> str:
    """The raw `status:` value, for a failure message that says why the loader
    dropped the page rather than making the reader go look."""
    match = re.search(r"^status:\s*(.+)$", text, re.M)
    return match.group(1).strip() if match else "<absent>"


def _description_as_written(text: str) -> str:
    """The `description:` value as a human reads the file: the key's text plus
    its indented continuation lines, joined the way YAML folds them."""
    block = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not block:
        return ""
    parts: list[str] = []
    taking = False
    for line in block.group(1).splitlines():
        if line.startswith("description:"):
            taking = True
            parts.append(line.split(":", 1)[1].strip())
        elif taking and line.startswith(" "):
            parts.append(line.strip())
        elif taking:
            break
    return " ".join(p for p in parts if p)


def test_no_skill_description_loses_its_tail_to_a_yaml_comment():
    """`#` preceded by a space opens a comment inside a plain scalar, so a
    description that mentions, say, backlog `#484` silently loses everything
    from the hash on — and `_parse_frontmatter` swallows every exception
    (`agent_mcp/skills.py:51-54`), so a page can also circulate with a
    half-sentence for a description and nothing raises.

    Found by the seam test above on this very page, whose served description
    read "…The supported fix is running under the venv interpreter; deleting
    the classes is backlog". Checked across every skill, because the cause is a
    YAML rule and not this page's; the fix on the page under #488 is the words
    "backlog item 484".
    """
    from agent_mcp import skills

    offenders = []
    pages = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    assert pages, f"{NO_VAULT} (no SKILL.md under {SKILLS_DIR})"
    for path in pages:
        content = _text(path)
        written = _description_as_written(content)
        if " #" not in written:
            continue
        served = (skills._parse_frontmatter(content)[0].get("description") or "")
        lost = written.split(" #", 1)[1]
        if lost.strip() and lost.strip() not in str(served):
            offenders.append(
                f"{path.parent.name}: description loses {lost.strip()[:60]!r} "
                f"as parsed; served {str(served)[-60:]!r}"
            )
    assert offenders == [], "\n".join(offenders)


def test_a_quarantine_status_pulls_the_page_from_retrieval(tmp_path):
    """The other half of the loader boundary, driven through the real loader on a
    fixture rather than re-implemented here: `_QUARANTINE_STATUSES` is what would
    make this skill vanish from `skills_search` while every file-level assertion
    above stayed green.

    This proves the mechanism the seam test's `is not None` claim rests on, that
    all five quarantine values still reach it, and that `active` is not itself one
    of them — if it were, every retrieval assertion in this file would be vacuous.
    It also pins the silent half: a *missing* `status` parses to `""` and is not
    quarantined, so a page whose front matter degraded still circulates, which is
    what the seam test's non-empty-description assert is for.
    """
    from agent_mcp import skills

    assert skills._load_skill(SKILLS_DIR / "autonomy-pipeline-yaml-fix") is not None
    assert "active" not in skills._QUARANTINE_STATUSES, (
        "`active` is in the quarantine set, so no skill on this box is retrievable "
        "and the assertions above prove nothing"
    )

    def write(tag: str, body: str) -> Path:
        d = tmp_path / f"skill-{tag}"
        d.mkdir()
        (d / "SKILL.md").write_text(body, encoding="utf-8")
        return d

    assert skills._load_skill(
        write("active", "---\nname: s\nstatus: active\ndescription: d\n---\n# B\n")
    ) is not None, "a status: active fixture did not load, so the live result is meaningless"
    for status in sorted(skills._QUARANTINE_STATUSES):
        assert skills._load_skill(
            write(status, f"---\nname: s\nstatus: {status}\ndescription: d\n---\n# B\n")
        ) is None, (
            f"status {status!r} no longer pulls a skill out of retrieval, so the "
            "quarantine this file's seam test guards has changed shape"
        )
    degraded = write("no-status", "---\nname: s\ndescription: d\n---\n# B\n")
    assert skills._load_skill(degraded) is not None, (
        "a page with no status key is now dropped, which means the seam test should "
        "be asserting presence rather than description"
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
