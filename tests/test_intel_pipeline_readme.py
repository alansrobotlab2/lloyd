"""#847 (finding 3): `scripts/intel-pipeline/README.md` documented a package that
did not exist — arXiv and Hacker News scanners, a `cd ~/obsidian/agents/...`
invocation into a vault tree deleted on 2026-09-03, and imports of modules
that were never in this checkout. Every `intel_pipeline.<module>` the README
names is resolved against the package on disk, and the dead paths are grepped
for, so the doc cannot describe a tree other than this one again.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "scripts" / "intel-pipeline"
README = PKG_DIR / "README.md"
PACKAGE = PKG_DIR / "intel_pipeline"


def _modules_named(text: str) -> set[str]:
    return set(re.findall(r"intel_pipeline(?:\.[A-Za-z_][A-Za-z0-9_]*)+", text))


def test_every_module_the_readme_names_exists_on_disk():
    named = _modules_named(README.read_text())
    assert named, "the README names no intel_pipeline module at all"
    for dotted in sorted(named):
        rel = Path(*dotted.split(".")[1:])
        assert (PACKAGE / rel).with_suffix(".py").exists() or (PACKAGE / rel).is_dir(), \
            f"README names {dotted}, which is not in {PACKAGE}"


def test_readme_architecture_tree_lists_the_real_scanners():
    text = README.read_text()
    on_disk = {p.stem for p in (PACKAGE / "scanners").glob("*_scanner.py")}
    listed = set(re.findall(r"([a-z_]+_scanner)\.py", text))
    assert listed == on_disk, (listed, on_disk)


def test_readme_points_at_the_live_invocation_not_the_deleted_vault_tree():
    text = README.read_text()
    assert "obsidian/agents" not in text
    assert "obsidian/memory/feeds" not in text
    assert "cd ~/lloyd/scripts/intel-pipeline" in text
    assert "python -m intel_pipeline" in text


# ── #3873: the scan coverage series is documented where the code writes it ───────────

#: Every key the writer emits, so the table cannot keep its prose while a field is renamed
#: out of it — which is what a README that only names the FILE would fail to catch.
SERIES_KEYS = ("youtube_feeds_fetched", "youtube_feeds_attempted", "youtube_degraded",
               "youtube_raw_rows", "items_written")


def test_the_readme_documents_the_scan_series_file_its_counts_and_the_command():
    """Clause 5: the file, the counts, and the ONE command are all named.

    Asserted against the writer's own constant rather than a spelled-out filename, so the
    README cannot pass by naming a file the code no longer writes. The command is checked
    as a substring of a fenced block because "look at the file" is the half of this the
    next reader of a blackout actually needs — #3873 was filed because the fraction existed
    only in eight transcripts whose wording differed run to run.
    """
    import sys
    text = README.read_text()
    pkg = PKG_DIR / "intel_pipeline"
    sys.path.insert(0, str(pkg.parent))
    from intel_pipeline import _paths

    name = _paths.SCAN_SERIES_FILE.name
    assert f"`{name}`" in text, f"the README never names {name}"
    for key in SERIES_KEYS:
        assert f"`{key}`" in text, f"the README omits the {key} count"
    assert "scan-series.jsonl" in text
    assert f"tail -n 14 ~/lloyd-data/_pipeline/vault-derived/memory/feeds/{name}" in text, (
        "no single command reads the curve")
    assert "append-only" in text.lower(), "the file's one property is undocumented"


# ── the stage-2 call budget: one number, stated where all three readers look ───
#
# `LLM_MAX_CALLS` is the number that decides how much of a day gets graded, and it is
# read by the reader of the README, the reader of the skill, and the code. On
# 2026-11-13 — the first full-coverage YouTube day since 2026-11-05 — a budget of 40
# asked 40 of the day's 129 survivors and left 89 `grade_source: call_cap`, all 9
# GitHub rows among them, so the GitHub half of the day wrote nothing. Nothing in the
# README or the skill said that was possible: the README stated `LLM_MAX_CALLS = 40`
# as a fact with no consequence attached, and the skill's Reality check still reported
# "7 of 7 survivors were model-graded" from 2026-09-15. These nodes keep the number and
# its measured cost in all three places, so a future budget change that updates only
# the code leaves two of them red. Same convention as
# tests/test_intel_pipeline_scorer.py's autonomy-task-30 nodes: fail when the vault is
# present and the claim is stale, skip only when the vault is genuinely absent.

SKILL = (Path.home() / "obsidian" / "skills"
         / "intelligence-pipeline" / "SKILL.md")


def _budget() -> int:
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))
    from intel_pipeline import scoring
    return scoring.LLM_MAX_CALLS


def _func_source(name: str) -> str:
    """The source segment of a top-level function in `scoring.py`, comments included."""
    path = PACKAGE / "scoring.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert fn is not None, f"scoring.py no longer defines a top-level {name}()"
    return ast.get_source_segment(src, fn)


def _section(text: str, start: str, end: str) -> str:
    """`text` between the first occurrence of `start` and the first `end` after it.

    A missing anchor is an assertion failure, not a `ValueError`: the anchor IS the
    claim (a README whose budget heading was renamed, or a skill whose Reality check
    was renumbered, has stopped stating what these nodes pin).
    """
    assert start in text, f"the text no longer contains the anchor {start!r}"
    i = text.index(start)
    rest = text[i + len(start):]
    assert end in rest, (
        f"nothing closes the section opened by {start!r} — no {end!r} follows it")
    j = rest.index(end)
    return text[i:i + len(start) + j]


def test_the_readme_states_the_budget_the_code_carries():
    """Clause 4: README's number == `scoring.LLM_MAX_CALLS`, no stale figure left."""
    cap = _budget()
    section = _section(README.read_text(encoding="utf-8"),
                       "#### Who gets a model call", "\n## ")

    assert f"LLM_MAX_CALLS = {cap}" in section, (
        f"the README's budget section does not state the constant's value {cap}")
    assert "LLM_MAX_CALLS = 40" not in README.read_text(encoding="utf-8") or cap == 40, \
        "the README still states the 40-call budget retired by #4117"


def test_the_readme_says_what_a_budget_below_the_day_costs():
    """Clause 4's other half: the number has a stated consequence, in plain words.

    The whole reason #4117 exists is that nothing written down said a budget below a
    day's survivor count can leave one source's entire day ungraded — the shape
    2026-11-13 hit with all 9 GitHub rows refused and nothing written for them.
    """
    section = _section(README.read_text(encoding="utf-8"),
                       "#### Who gets a model call", "\n## ")

    assert "survivor" in section and "ungraded" in section, (
        "the budget section states the number but not what a budget below the day's "
        "survivor count costs")
    sentences = [s for s in section.replace("\n", " ").split(". ")
                 if "survivor" in s and "ungraded" in s]
    assert sentences, "no single sentence names both the survivors and the consequence"
    assert any("source" in s for s in sentences), (
        "the consequence must name that a whole source's day can go ungraded: "
        f"{sentences}")
    assert "2026-11-13" in section, "the section cites no measured day"


def test_the_scorer_module_no_longer_calls_the_caps_cost_unmeasured():
    """Clause 4: the comment inside `stage2_score` carries the measured number.

    Read off the function's own source segment, not the file, so the same stale
    sentence sitting in some other docstring cannot pass or fail this one.
    """
    body = _func_source("stage2_score")

    for stale in ("unmeasured", "0 `grade_source: call_cap` rows",
                  "70 of 70 graded rows model-graded"):
        assert stale not in body, (
            f"`stage2_score` still claims the cap's cost is {stale!r}:\n"
            + "\n".join(line for line in body.splitlines() if stale in line))
    for measured in ("89", "129", "2026-11-13", "0.76", "1800"):
        assert measured in body, (
            f"the comment never names the measured {measured} — the refused count, "
            "the survivor count, the day, the per-call cost and the task timeout the "
            "budget is sized against are all four of a kind")


def test_the_skill_states_the_budget_as_a_number_and_the_measured_day():
    """Clause 5: §3 gives the budget as a number and names the day shape; Reality
    check item 4 carries the 2026-11-13 measurement instead of 2026-09-15's 7 of 7.
    """
    if not SKILL.exists():
        pytest.skip(f"no vault checkout at {SKILL.parent}")
    cap = _budget()
    text = SKILL.read_text(encoding="utf-8")

    scoring_step = _section(text, "3. **Scoring**", "\n4. **Vault writer**")
    assert "capped at 40 calls/run" not in scoring_step, (
        "§3 still states the retired budget as a word-form cap")
    assert str(cap) in scoring_step, (
        f"§3 does not give the budget as the number {cap}")
    assert "full-coverage" in scoring_step, (
        "§3 never names the full-coverage YouTube day as the shape that stresses the "
        "budget — the first thing a run hits on one")

    reality_item = _section(text, "4. **`LLM_KEYWORD_THRESHOLD", "\n5. **")
    assert "7 of 7 survivors were model-graded" not in reality_item, (
        "Reality check item 4 still reports 2026-09-15's 7 of 7")
    for measured in ("40 of 129", "89", "call_cap", "9 GitHub"):
        assert measured in reality_item, (
            f"Reality check item 4 never names the measured {measured}: {reality_item}")
    assert "40-call" not in text, (
        "the skill still states the 40-call budget somewhere:\n"
        + "\n".join(line for line in text.splitlines() if "40-call" in line))
