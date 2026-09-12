"""``knowledge/_log.md`` is one shape, has one owner, and an append cannot re-break it (#873).

The file is the only change history Lloyd has, and in September 2026 it was simultaneously
unparseable and unowned: two entries shared one line
(``…gaussian-splatting-vs-nerf.md## [2026-06-19] research | Real-time depth estimation…``), so
``grep -o "## " | wc -l`` said 10 while ``grep -c "^## "`` said 9; the dates were not newest-first;
every heading was ``## [YYYY-MM-DD]`` where OKF §9 requires bare ISO ``YYYY-MM-DD``; and
``KNOWLEDGE_SCHEMA.md`` told *every* writer to append without naming an owner, which is how it
accumulated 10 entries against 342 commits that touched ``knowledge/``. Four skill templates
handed out the broken heading, so the next writer produced the defect on purpose.

The repair and the contract are one change, so this file pins both: the shape of the file
(clauses 1-4), the death of the bad template and the unowned rule (5-6), one scheduled owner
with a real logged run behind it (7-8), and the property that made the whole thing worth doing —
an append that cannot re-create the collision (9).

The vault half of this is markdown; ``scripts/vault/knowledge_log.py`` is the code half. The two
meet at a ``Bash`` line inside a skill file, which no import graph can see, so
``test_the_command_the_skill_runs_keeps_the_counts_equal`` runs that command for real rather than
trusting the prose around it.
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

from scripts.vault.knowledge_log import (  # noqa: E402
    DEFAULT_LOG,
    LogShapeError,
    append_entry,
    check,
    read_entries,
)

VAULT = Path.home() / "obsidian"
LOG = VAULT / "knowledge" / "_log.md"
SCHEMA = VAULT / "knowledge" / "KNOWLEDGE_SCHEMA.md"
HELPER = ROOT / "scripts" / "vault" / "knowledge_log.py"
DATE_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2})$")
#: Strings the file carried before the repair; losing one means the rewrite dropped an entry.
PRESERVED = (
    "gaussian-splatting-vs-nerf.md",
    "Real-time depth estimation",
    "slam-vla-integration.md",
    "fastvlm-apple-edge-vlm.md",
    "2026-09-08-deepseek-v4-1-flash-beta.md",
    "2214 pages updated",
)


def skill_files() -> list[Path]:
    """The live skill templates — the same set the acceptance greps address."""
    return sorted((VAULT / "skills").glob("*/SKILL.md")) + sorted((ROOT / "skills").glob("*/SKILL.md"))


def counts(text: str) -> tuple[int, int]:
    """Clause 1's two measurements: every ``'## '`` occurrence, and every line starting with one."""
    return text.count("## "), sum(1 for line in text.splitlines() if line.startswith("## "))


def date_headings(text: str) -> list[str]:
    return [m.group(1) for m in (DATE_HEADING.match(line) for line in text.splitlines()) if m]


def scheduled_skill_names() -> dict[str, Path]:
    """``skill_name:`` in a task file is the only thing that makes a skill a scheduled writer."""
    out: dict[str, Path] = {}
    for task in sorted((VAULT / "autonomy").glob("*.md")):
        match = re.search(r"^skill_name:\s*(\S+)\s*$", task.read_text(encoding="utf-8"), re.M)
        if match:
            out[match.group(1)] = task
    return out


def owners_of_the_log() -> list[Path]:
    return [path for path in skill_files() if "_log.md" in path.read_text(encoding="utf-8")]


def operations_log_section() -> str:
    text = SCHEMA.read_text(encoding="utf-8")
    start = text.find("## Operations Log")
    assert start != -1, "KNOWLEDGE_SCHEMA.md lost its Operations Log section entirely"
    rest = text[start + len("## Operations Log"):]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


# --- clause 1: one entry per line ---------------------------------------------------------


def test_heading_counts_agree_on_the_live_log():
    raw = LOG.read_text(encoding="utf-8")
    occurrences, line_initial = counts(raw)
    assert occurrences == line_initial, (
        f"{occurrences} '#{'#'} ' occurrences vs {line_initial} heading lines: an entry shares a "
        "line with another, so every per-line parser sees fewer entries than exist"
    )


def test_check_agrees_that_the_live_log_is_well_formed():
    assert check(LOG) == [], "the shape checker and the live file disagree:\n" + "\n".join(check(LOG))


# --- clause 2: ISO date headings ----------------------------------------------------------


def test_no_bracketed_date_heading_survives_in_the_log():
    offenders = [line for line in LOG.read_text(encoding="utf-8").splitlines() if line.startswith("## [")]
    assert offenders == [], f"OKF §9 wants '## YYYY-MM-DD', found: {offenders}"


# --- clause 3: newest first ---------------------------------------------------------------


def test_the_log_reads_newest_first():
    dates = date_headings(LOG.read_text(encoding="utf-8"))
    assert dates, "no ISO date heading at all"
    inversions = [f"{dates[i]} above {dates[i + 1]}" for i in range(len(dates) - 1) if dates[i] < dates[i + 1]]
    assert inversions == [], f"dates must be non-increasing top-to-bottom, found: {inversions}"
    assert dates[0] == max(dates), f"the newest entry must be the first heading, found {dates[0]} above {max(dates)}"


# --- clause 4: the repair lost nothing ----------------------------------------------------


def test_the_repair_lost_no_entry():
    raw = LOG.read_text(encoding="utf-8")
    headings = date_headings(raw)
    assert len(headings) >= 10, f"the log held 10 entries before the repair and now has {len(headings)}"
    for needle in PRESERVED:
        assert needle in raw, f"entry text disappeared from the repair: {needle!r}"


# --- clause 5: the bad template is dead ---------------------------------------------------


def test_no_skill_template_prescribes_the_bracketed_heading():
    offenders = [
        f"{path}:{line}"
        for path in skill_files()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "## [YYYY-MM-DD]" in line
    ]
    assert offenders == [], f"a writer handed the broken heading would reproduce the defect: {offenders}"


# --- clause 6: the schema rule is owned, not broadcast ------------------------------------


def test_the_schema_names_one_owner():
    section = operations_log_section()
    assert "should append an entry to" not in section, (
        "the rule that told every writer to append is still standing; it produced 10 entries "
        "against 342 commits and must name an owner instead"
    )
    named = {skill.parent.name for skill in skill_files() if skill.parent.name in section}
    assert named == {"nightly-reflection-knowledge-write"}, (
        f"the append contract needs exactly one owner named, found: {sorted(named) or 'nobody'}"
    )


# --- clause 7: that owner is scheduled, and nothing else prescribes the file --------------


def test_the_log_contract_has_one_scheduled_owner_that_appends_through_the_helper():
    prescribers = owners_of_the_log()
    assert prescribers, "no skill mentions _log.md and the schema still points at it: who owns it?"
    scheduled = scheduled_skill_names()
    unscheduled = sorted(skill.parent.name for skill in prescribers if skill.parent.name not in scheduled)
    assert unscheduled == [], (
        f"skills that tell a writer to append but run on no schedule: {unscheduled} — that is "
        "the unowned-rule defect #453 recorded, in template form"
    )
    for skill in prescribers:
        text = skill.read_text(encoding="utf-8")
        assert "knowledge_log.py append" in text, f"{skill} mentions the log but never calls the helper"


# --- clause 8: the owner has actually logged a run it recorded ----------------------------


def test_the_owner_logs_the_runs_it_records():
    owner = owners_of_the_log()[0]
    task = scheduled_skill_names()[owner.parent.name]
    body = task.read_text(encoding="utf-8")
    logged = set(date_headings(LOG.read_text(encoding="utf-8")))
    runs: dict[str, list[str]] = {}
    for run_id, stamp in re.findall(r"Run (run_\d+_(\d{8})_\d+) — success", body):
        runs.setdefault(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}", []).append(run_id)
    assert runs, f"{task.name} records no successful run to test against"

    entries: dict[str, str] = {}
    for date, entry_body in read_entries(LOG):
        entries[date] = f"{entries.get(date, '')} {entry_body}"

    # The entry must name the run it describes, not merely land on a day the task happened to
    # run: a pre-contract note with a coincidental date proves nothing.
    named = [(day, run_id) for day in sorted(logged, reverse=True) for run_id in runs.get(day, [])
             if run_id in entries.get(day, "")]
    assert named, (
        f"_log.md holds no entry dated the day of a successful {task.name} run that names that "
        f"run; the newest days the owner ran are {sorted(runs)[-3:]} and the log's newest "
        f"heading is {max(logged)} — the owner is appending, or it is not"
    )
    day, run_id = named[0]
    run_log = ROOT / "autonomy-runs" / task.stem.split("-")[0] / f"{run_id}.md"
    assert run_log.exists() or not run_log.parent.exists(), f"{run_id} is claimed but its run log is missing"


# --- clause 9: an append cannot re-create the collision -----------------------------------


@pytest.fixture
def log_copy(tmp_path):
    """A real copy of the live file — the append is exercised against the shape that broke."""
    target = tmp_path / "_log.md"
    target.write_text(LOG.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_one_append_cannot_recreate_the_collision(log_copy):
    before = counts(log_copy.read_text(encoding="utf-8"))[0]
    assert append_entry(log_copy, "run run_39_20260912_070000 — knowledge write: 1 note (knowledge/software/x.md)",
                        kind="nightly-reflection-knowledge-write", day="2026-09-12") == "## 2026-09-12"
    assert append_entry(log_copy, "lint probe", kind="lint", day="2024-01-05") == "## 2024-01-05"

    raw = log_copy.read_text(encoding="utf-8")
    occurrences, line_initial = counts(raw)
    assert occurrences == line_initial == before + 2, "an append changed one count but not the other: two entries share a line"
    assert "md## [" not in raw and "## [" not in raw, "the append re-introduced the bracketed collision form"
    assert raw.endswith("\n") and not raw.endswith("\n\n"), "the file must end in exactly one newline or the next append collides"
    dates = date_headings(raw)
    assert dates == sorted(dates, reverse=True), "the append put an entry out of newest-first order"
    assert check(log_copy) == []


def test_the_command_the_skill_runs_keeps_the_counts_equal(log_copy):
    """The seam: a Bash line in a SKILL.md reaches this repo across a process boundary."""
    result = subprocess.run(
        [sys.executable, str(HELPER), "append",
         "--kind", "nightly-reflection-knowledge-write",
         "--text", "run run_39_20260912_070000 — knowledge write: 2 notes (knowledge/software/a.md, knowledge/software/b.md)",
         "--date", "2026-09-12", "--log", str(log_copy)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    raw = log_copy.read_text(encoding="utf-8")
    assert counts(raw)[0] == counts(raw)[1], f"the CLI wrote a colliding entry: {result.stdout}"
    assert raw.endswith("\n") and "## 2026-09-12" in raw

    verify = subprocess.run([sys.executable, str(HELPER), "check", "--log", str(log_copy)],
                            capture_output=True, text=True, timeout=60)
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_a_value_that_would_open_a_second_heading_is_refused_not_written(log_copy):
    before = log_copy.read_text(encoding="utf-8")
    with pytest.raises(LogShapeError):
        append_entry(log_copy, "fine ## [2099-01-01] smuggled", day="2026-09-12")
    with pytest.raises(LogShapeError):
        append_entry(log_copy, "fine", kind="## 2099-01-01", day="2026-09-12")
    with pytest.raises(LogShapeError):
        append_entry(log_copy, "fine", day="09-12-2026")
    assert log_copy.read_text(encoding="utf-8") == before, "a refused append still touched the file"


def test_re_appending_a_recorded_entry_does_not_double_it(log_copy):
    text = "run run_39_20260912_070000 — knowledge write: 1 note"
    assert append_entry(log_copy, text, kind="lint", day="2026-09-12") == "## 2026-09-12"
    assert append_entry(log_copy, text, kind="lint", day="2026-09-12") is None, (
        "a retry of a run that already logged must not add a second entry for it"
    )


# --- clause 9's other half: no second writer is left --------------------------------------


def test_the_lint_migrator_no_longer_writes_the_log():
    """``knowledge-frontmatter-backfill.py`` was the other hand on this file."""
    src = (ROOT / "scripts" / "knowledge-frontmatter-backfill.py").read_text(encoding="utf-8")
    assert "LOG_FILE" not in src, "the migrator still holds its own handle on the log"
    # Quoted occurrences are code; a comment may name the file. Exactly one code reference is
    # allowed, and it is the SKIP_FILES entry that stops the migrator rewriting the log itself.
    assert src.count('"_log.md"') == 1, (
        "the only code reference to _log.md must be the SKIP_FILES entry that keeps it unmigrated"
    )
    assert "SKIP_FILES" in src


def test_the_default_target_is_the_live_vault_log():
    assert DEFAULT_LOG == LOG, (
        "an owner that wrote somewhere else would look like a working contract while the real "
        "log kept rotting; the default is not configurable by environment for that reason"
    )


# --- these tests can fail: the checker against the shape the file actually had -----------


def test_check_names_every_defect_the_log_had_before(tmp_path):
    broken = "\n".join([
        "# Knowledge Log",
        "",
        "## [2026-06-18] quick-research | 3D Gaussian Splatting vs NeRF — 6 facts,"
        "knowledge/robotics/gaussian-splatting-vs-nerf.md## [2026-06-19] research | Real-time"
        " depth estimation — updated existing note",
        "## [2026-07-09] quick-research | Real-time policy adaptation — 7 facts,"
        "knowledge/robotics/real-time-policy-adaptation-robots.md",
        "## [2026-06-14] medium-research | Multi-agent coordination — 7 sources,"
        "knowledge/robotics/multi-agent-coordination.md",
    ]) + "\n"
    target = tmp_path / "_log.md"
    target.write_text(broken, encoding="utf-8")
    problems = "\n".join(check(target))
    assert "occurrences vs" in problems, "the checker cannot see two entries sharing a line"
    assert "bracketed date heading" in problems, "the checker cannot see the §9 violation"
    assert "no ISO date heading" in problems, "the checker accepts a file with no conformant heading at all"


def test_a_missing_trailing_newline_is_reported(tmp_path):
    target = tmp_path / "_log.md"
    target.write_text("# Knowledge Log\n\n## 2026-09-01\nlint | older\n## 2026-09-11\nlint | newer\n", encoding="utf-8")
    problems = "\n".join(check(target))
    assert "newest-first is broken" in problems, "an inverted file must not read as healthy"
    assert "newest entry is 2026-09-01 but 2026-09-11 exists" in problems
    target.write_text(target.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    assert any("does not end with a newline" in problem for problem in check(target))


def check_path_text(text: str) -> list[str]:
    """Run the checker over a string by way of a temp file, same code path as the live log."""
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        target = Path(work) / "_log.md"
        target.write_text(text, encoding="utf-8")
        return check(target)
