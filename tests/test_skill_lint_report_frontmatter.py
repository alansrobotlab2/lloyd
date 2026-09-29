"""#1826 — the skill-lint report declares its own OKF type at birth.

`scripts/skill_lint.py::main` writes `~/obsidian/autonomy/skill-lint-report.md`
with `REPORT_PATH.write_text(render_report(result))`, and `render_report` returns a
body whose first line is the `# Skill Lint Report — <ts>` heading
(`scripts/skill_lint.py:1527`, `:1840`). No frontmatter, ever: `git log --oneline
-S'frontmatter' -- scripts/skill_lint.py` names only the *skill*-side parser. So the
file is a permanent row in `scripts/vault/validate_okf.py`'s violation list
(`autonomy/skill-lint-report.md: no parseable frontmatter block`), and the two times
a human typed it by hand (vault `8b064ccb` 09-18, `87d8f7db` 09-19) the next lint
firing erased it again — a writer that overwrites a file owns that file's frontmatter.

The fix declares it at birth. What is asserted here is the *bytes on disk after
`main()` returns*, never a rendering in memory, because the consumer
(`validate_okf.py`) reads the file and nothing reads the string.

Two seams are crossed for real, not mirrored:

  * skill_lint → vault bytes → `validate_okf.py`: the gate's own strict regex and
    its own stranded-block detector, imported from the modules the weekly gate
    imports, plus the gate's own command line (`--root`) over a scratch vault, so a
    block this writer emits that the gate would reject cannot pass here.
  * the nightly's invocation (`python scripts/skill_lint.py` under a redirected
    `$HOME`) → those bytes → the gate, end to end in two subprocesses.

Everything runs against `tmp_path` or a scratch `$HOME`. Nothing here reads or
writes the live vault: the acceptance run against `~/obsidian` is a real-skill-lint
run and stays owed after landing (item #1826), because a fixture cannot prove the
live file's next write carries the block — only the next real run can.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The real gate's objects, not a copy of them: a writer that produces a block the
# strict rule rejects must fail here even where a lenient parser would forgive it.
from scripts.vault.validate_okf import STRICT_FM_RE  # noqa: E402
from scripts.vault.okf_stranded import find_stranded_frontmatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: A finding-set result of the shape `tests/test_fact_identity_one_action_one_fact.py`
#: fabricates: one skill, one MISSING_SCRIPT finding, so the body has a table in it
#: and the frontmatter is graded against a real report and not an empty one.
RESULT = {
    "generated_at": "2026-09-29T01:45:33",
    "total": 1,
    "dead": [],
    "missing_desc": [],
    "drift": [],
    "duplicates": [],
    "stale": [],
    "phantom": [],
    "missing_script": [{"name": "demo-skill", "path": "/x/SKILL.md",
                        "scripts": [{"path": "scripts/demo.py", "known_stale": ""}]}],
}

#: `^---` on its own line — the fence count clause 2 is about. The report's markdown
#: tables use `|---|`, which is not this, which is why one emitted block cannot
#: leave a second fence anywhere (item #1826 triage: `grep -c "^---$"` → 0).
_FENCE_LINE = re.compile(r"^---[ \t]*$", re.MULTILINE)


def _load_skill_lint():
    """The script as a module, the way the other skill-lint tests load it."""
    spec = importlib.util.spec_from_file_location(
        "skill_lint", ROOT / "scripts" / "skill_lint.py")
    skill_lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill_lint)
    return skill_lint


@pytest.fixture
def report_bytes(tmp_path, monkeypatch):
    """The bytes `main()` writes, into `tmp_path/obsidian/autonomy/`.

    `REPORT_PATH` is patched rather than `$HOME`, so the live vault is never in
    reach; the tree is shaped like a vault (`obsidian/autonomy/`) because `main()`
    derives the git sample from `REPORT_PATH.parent.parent` and a flat tmp dir
    simply skips that step.
    """
    skill_lint = _load_skill_lint()
    report_path = tmp_path / "obsidian" / "autonomy" / "skill-lint-report.md"
    monkeypatch.setattr(skill_lint, "REPORT_PATH", report_path)
    monkeypatch.setattr(skill_lint, "lint", lambda: RESULT)
    assert skill_lint.main() == 0, "main() must still exit 0 (advisory-only script)"
    return report_path.read_text(encoding="utf-8")


def _fm(text: str) -> dict:
    """The leading block, parsed the way `validate_okf.main` parses it."""
    m = STRICT_FM_RE.match(text)
    assert m, f"no parseable frontmatter block; first line is {text.splitlines()[0]!r}"
    fm = yaml.safe_load(m.group(1))
    assert isinstance(fm, dict), f"frontmatter is not a mapping: {fm!r}"
    return fm


# --- clause 1: the written bytes carry a strict block with the declared keys ----

def test_main_writes_a_strict_frontmatter_block(report_bytes):
    """Clause 1: `main()`'s bytes parse, and carry type/segment/tags.

    The three keys are the ones the item pins. `type` is the only one
    `validate_okf` gates on (`scripts/vault/validate_okf.py` fails a missing or
    empty `type`), so the assertion that matters most is that it is non-empty and
    in `okf_taxonomy.KNOWN_TYPES`; `segment` and `tags` are convention here, and
    they are pinned anyway because the item names them and #1692's owed entry
    agrees on the values.
    """
    assert report_bytes.startswith("---\n"), "the block must be the first thing on disk"
    fm = _fm(report_bytes)
    assert fm["type"] == "note", f"type is {fm['type']!r}, expected 'note'"
    assert fm["segment"] == "autonomy"
    assert isinstance(fm["tags"], list) and fm["tags"], f"tags empty: {fm.get('tags')!r}"
    assert "skill-lint" in fm["tags"]

    from scripts.vault.okf_taxonomy import KNOWN_TYPES
    assert str(fm["type"]).strip() in KNOWN_TYPES, (
        "`type` would pass validate_okf but warn as unknown vocabulary")


# --- clause 2: exactly one block, so validate_okf's stranded detector stays quiet -

def test_the_written_bytes_hold_exactly_one_frontmatter_block(report_bytes):
    """Clause 2: no second `---`-delimited block stranded below the emitted one.

    Both of the gate's two answers are asserted: its fence-level view (exactly two
    `^---$` lines, the block's own pair) and the structural detector
    `validate_okf.main` and `okf_migrate` share, which is what actually turns a
    stranded body block into a violation (#960). A writer that prepends to a body
    that already had a block would satisfy the fence count only by luck.
    """
    assert len(_FENCE_LINE.findall(report_bytes)) == 2, (
        "the emitted block plus exactly zero more; a third fence line means the "
        "header was appended to a body that already had one")
    assert len(STRICT_FM_RE.findall(report_bytes)) == 1
    assert find_stranded_frontmatter(report_bytes) == [], (
        "validate_okf's stranded detector fires on the report this writer emits")


# --- clause 3: summary present, timestamp equal to the heading it sits above ----

def test_the_block_carries_a_summary_and_the_heading_timestamp(report_bytes):
    """Clause 3: the stamp is the run's own, not a stale one.

    `render_report` takes its heading timestamp from `result["generated_at"]`
    (`scripts/skill_lint.py:1501`, `:1527`), and the block has to agree with it or
    a later reader has two timestamps and no way to tell which is the run. So the
    equality is against the heading line as written on disk, immediately after the
    block — not against a recomputed `now()`, which would be a second source.
    """
    fm = _fm(report_bytes)
    assert str(fm.get("summary", "")).strip(), "summary missing or empty"

    m = STRICT_FM_RE.match(report_bytes)
    heading = report_bytes[m.end():].splitlines()[0]
    ts = heading.split("—")[-1].strip()
    assert heading.startswith("# Skill Lint Report — "), (
        f"the body must still open with its heading, immediately below the block: {heading!r}")
    assert str(fm["timestamp"]) == ts, (
        f"frontmatter {fm['timestamp']!r} != heading {ts!r} from {RESULT['generated_at']!r}")
    assert ts == RESULT["generated_at"]


# --- the gate itself, over the bytes it will be pointed at ----------------------

def _run_gate(vault: Path) -> subprocess.CompletedProcess:
    """The live acceptance command line, pointed at a scratch tree."""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vault" / "validate_okf.py"),
         "--root", str(vault)],
        capture_output=True, text=True, timeout=120)


def test_validate_okf_prints_no_row_for_the_report_it_is_handed(tmp_path, report_bytes):
    """`validate_okf.py --root` over a scratch vault holding just this report.

    The live acceptance check is the script's own command line, so the script's own
    command line is run here. Passing `--root` is what makes it a fixture rather
    than the live vault.

    Run twice, because the absence of the row is the assertion and a clean exit is
    also what a scan that never reached the file prints: `validate_okf` names a path
    only when it violates. The control half grades the same tree with the emitted
    header cut back off — the pre-fix bytes — and must produce the row verbatim; only
    then does the second run's silence say anything about the header rather than about
    the walk.
    """
    vault = tmp_path / "obsidian"
    report = vault / "autonomy" / "skill-lint-report.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    m = STRICT_FM_RE.match(report_bytes)
    assert m, "the fixture bytes have no header to cut off"
    report.write_text(report_bytes[m.end():], encoding="utf-8")
    control = _run_gate(vault)
    assert control.returncode == 1, (
        "the gate is clean over a report with NO frontmatter, so the scan is not "
        f"reaching this tree at all:\n{control.stdout}{control.stderr}")
    assert "autonomy/skill-lint-report.md: no parseable frontmatter block" in control.stdout, (
        "the control row the live vault prints today is not the row this scan "
        f"prints here:\n{control.stdout}")

    report.write_text(report_bytes, encoding="utf-8")
    proc = _run_gate(vault)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"validate_okf exited {proc.returncode}:\n{out}"
    assert "VIOLATIONS : 0" in out, out
    assert "skill-lint-report.md" not in out.split("VIOLATIONS")[1], out


# --- the nightly's own invocation, end to end under a redirected $HOME ----------

def test_the_nightly_invocation_writes_a_conformant_report(tmp_path):
    """`python scripts/skill_lint.py` with `$HOME` in `tmp_path`, then the gate.

    This is the process boundary the fix actually lives across: the nightly job
    runs the script by path in its own interpreter (`tests/test_skills_single_walk.py`
    pins that boot shape), and `REPORT_PATH` follows `$HOME`, so redirecting it is
    what keeps the live vault out of a test — and it is why this node is cheap: the
    skill roots are `$HOME`-derived too, so the run scans 0 skills and writes a
    zero-finding report in under two seconds. What that run still has to get right
    is the thing under test — the real entry point, the real `main()`, the real
    write, no fabricated `REPORT_PATH` — which the fixture above cannot prove, since
    it patches the path and `lint` out.
    """
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "skill_lint.py")],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Totals:" in proc.stdout, proc.stdout[-1000:]

    report = tmp_path / "obsidian" / "autonomy" / "skill-lint-report.md"
    assert report.is_file(), "the nightly's report path moved"
    text = report.read_text(encoding="utf-8")
    fm = _fm(text)
    assert str(fm["type"]).strip() == "note"
    assert str(fm.get("summary", "")).strip()
    assert find_stranded_frontmatter(text) == []

    # The gate run is the redundancy here, not the evidence — the control that proves
    # the scan reaches this tree at all lives in
    # `test_validate_okf_prints_no_row_for_the_report_it_is_handed`, and `_fm` above
    # already proves the header is in these bytes.
    gate = _run_gate(tmp_path / "obsidian")
    assert gate.returncode == 0, gate.stdout + gate.stderr
    assert "VIOLATIONS : 0" in gate.stdout, gate.stdout
