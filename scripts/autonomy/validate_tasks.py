#!/usr/bin/env python3
"""validate_tasks.py — fail-loud linter for ~/obsidian/autonomy task files.

The autonomy scheduler (autonomy._parse_task_file) silently drops any task whose
frontmatter fails yaml.safe_load — the task vanishes from the schedule with no
failure record and no alert. On 2026-05-28 a bulk edit corrupted the `tags` field
of 34/40 task files (inline list followed by orphan block-list items), which
dormant-killed ~85% of the autonomy system for six days before anyone noticed.

This script makes that failure mode loud. Run it in CI / a healthcheck / before
any bulk edit of the autonomy dir. Exit codes:
    0  all task files parse and pass structural checks
    1  one or more task files are UNPARSEABLE (scheduler-invisible) — critical
    2  files parse but have structural problems (bad depends_on, no skill, ...)

Beyond parseability it checks that a field's value *resolves*, not just that the
key is present (#811) — three classes the scheduler only objects to at dispatch:

  * `depends_on` naming a task that parses but is parked (status outside
    up_next/in_progress). The dispatch gate now *sees* a paused upstream
    (`autonomy.dependency_resolution_set()`, #870) and holds the dependent, which
    looks identical to "the chain is merely late" from outside.
  * `model:` outside the `models:` keys and declared aliases.
    `autonomy._get_model_env` answers an unmatched name with `{}`, so the run goes
    to the primary's endpoint under a name it does not serve and the engine
    replies 404 — shaped like an engine being down, which
    `_record_failure(kind="infra")` keeps off the retry budget.
  * `skill_name`/`skill_path` that resolves to no SKILL.md under the skills dirs,
    which fails at dispatch with "Skill not found".

Usage:
    python scripts/autonomy/validate_tasks.py [--autonomy-dir DIR] [--strict]
        [--skills-dir DIR ...] [--config FILE]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# The scheduler's own frequency vocabulary (#815): a `frequency:` outside it
# with no `runs_per_day` resolves to no interval and never dispatches, silently
# until 2026-09-24. Imported rather than restated so the linter cannot drift
# from the map it lints against. `None` means the check could not run.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from autonomy import FREQUENCY_INTERVALS  # noqa: E402
except Exception:  # noqa: BLE001 — a linter with no scheduler beside it
    FREQUENCY_INTERVALS = None
# The grader's own reading of an `acceptance:` block (#623), for the same reason.
try:
    from app.run_acceptance import acceptance_problems  # noqa: E402
except Exception:  # noqa: BLE001
    acceptance_problems = None

DEFAULT_DIR = Path.home() / "obsidian" / "autonomy"
# Mirrors the slug branch of autonomy._load_skill_content (:894), which is the
# only skills root that path consults. --skills-dir overrides it.
DEFAULT_SKILLS_DIR = Path.home() / "obsidian" / "skills"
# config.yaml sits at the repo root, two levels above scripts/autonomy/.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
# The KNOWN-STATUS vocabulary, not a runnability set: it holds paused, draft,
# archived, done and failed too, and is used only for the `unknown status`
# warning below. Writing `if dep_status not in RUNNABLE_STATUSES` for the
# dependency check produces a check that can never fire (#811).
RUNNABLE_STATUSES = {"up_next", "in_progress", "draft", "paused", "archived", "done",
                     # `failed` = disabled after max_retries consecutive
                     # failures; a human re-enables it by setting up_next.
                     "failed", "archived"}
# The statuses an upstream may be in for a dependent to have any chance of
# dispatching. Deliberately narrower than autonomy._all_runnable_tasks, which
# also keeps `failed` so a disabled upstream stays *findable* by the dependency
# gate. Findable is not runnable: a `failed` upstream still parks its
# dependents, so it still warns here.
DEPENDABLE_STATUSES = {"up_next", "in_progress"}
NULLISH = ("", "null", "none")
_UNSET = object()


def _clean(value) -> str:
    """A frontmatter value as a comparable string, '' for null/None."""
    return str(value if value is not None else "").strip()


def _is_nullish(value: str) -> bool:
    return value.strip().lower() in NULLISH


def _parse(path: Path):
    """Mirror autonomy._parse_task_file: split on '---\\n' and yaml.safe_load."""
    content = path.read_text(encoding="utf-8")
    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return None, "no frontmatter (need leading '---' block)"
    try:
        fm = yaml.safe_load(parts[1])
    except Exception as e:  # noqa: BLE001 — we want the raw yaml error text
        first = str(e).splitlines()[0]
        return None, f"yaml parse error: {first}"
    if not isinstance(fm, dict):
        return None, "frontmatter is not a mapping"
    return fm, None


def model_names(config_path: Path) -> set[str] | None:
    """Every `model:` value the engine will accept: keys of `models:` + aliases.

    Mirrors autonomy._get_model_env (:1141-1155): it matches the name as a key of
    `models:`, then as any entry's `alias`, then returns {}. So a value outside
    this set is exactly a value that reaches the endpoint unrecognized.

    Returns None when no `models:` mapping can be read — the caller must treat
    that as "cannot check", never as "nothing to warn about".
    """
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        return None
    names = set(str(k) for k in models)
    for cfg in models.values():
        if isinstance(cfg, dict):
            alias = _clean(cfg.get("alias"))
            if alias:
                names.add(alias)
    return names


def skill_resolves(value: str, skills_dirs: list[Path]) -> bool:
    """Whether autonomy._load_skill_content would return content for this value.

    A value containing '/' or ending in .md is used as a path and nothing else —
    a missing path returns None rather than falling back to the slug branch
    (:883-891). Anything else is a slug resolved against each skills dir as
    `<dir>/<slug>/SKILL.md` (:894). A path naming a directory fails the same way
    dispatch does: read_text on it raises, and the loader returns None.
    """
    expanded = Path(value.replace("~", str(Path.home())))
    if "/" in value or value.endswith(".md"):
        return expanded.is_file()
    for d in skills_dirs:
        if (d / expanded / "SKILL.md").is_file():
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autonomy-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--skills-dir", action="append", default=None,
                    metavar="DIR",
                    help="directory of <slug>/SKILL.md to resolve skill_name "
                         "against; repeatable. Default: ~/obsidian/skills, the "
                         "only skills root the dispatch loader consults.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="yaml whose `models:` keys and aliases define the "
                         "accepted model: values")
    ap.add_argument("--strict", action="store_true",
                    help="treat structural warnings as failures (exit 2)")
    args = ap.parse_args()

    d = Path(args.autonomy_dir).expanduser()
    if not d.is_dir():
        print(f"error: {d} is not a directory", file=sys.stderr)
        return 2

    skills_dirs = [Path(s).expanduser() for s in (args.skills_dir or [DEFAULT_SKILLS_DIR])]
    config_path = Path(args.config).expanduser()

    # Task files follow the NN-name.md convention. Other .md files in the dir
    # (reports, notes) are ignored by the scheduler because _parse_task_file
    # returns None for them; we mirror that by only linting numbered files.
    files = sorted(
        (p for p in d.glob("*.md") if re.match(r"\d+-", p.name)),
        key=lambda p: int(re.match(r"\d+", p.name).group()),
    )

    unparseable: list[tuple[str, str]] = []
    parsed: dict[str, dict] = {}      # id -> fm
    by_file: list[tuple[Path, dict]] = []

    for p in files:
        fm, err = _parse(p)
        if err:
            unparseable.append((p.name, err))
            continue
        by_file.append((p, fm))
        tid = str(fm.get("id", "")).strip()
        if tid:
            parsed[tid] = fm

    # Read once, and only if some file declares a model. model_names returns None
    # when no `models:` mapping can be read; that must never read as a clean pass,
    # so it becomes a loud per-file warning below instead of silence.
    known_models = _UNSET

    warnings: list[str] = []
    for p, fm in by_file:
        status = str(fm.get("status", "") or "").strip()
        if status and status not in RUNNABLE_STATUSES:
            warnings.append(f"{p.name}: unknown status '{status}'")
        if status in ("up_next", "in_progress"):
            skill = str(fm.get("skill_name", "") or "").strip()
            spath = str(fm.get("skill_path", "") or "").strip()
            if not skill and not spath:
                warnings.append(f"{p.name}: runnable but has no skill_name/skill_path")
        # Existence, not just presence: a name that resolves to no SKILL.md fails
        # at dispatch with "Skill not found" (autonomy.py:1435). Checked for every
        # status, not just runnable ones — a parked task is precisely where a bad
        # value waits unnoticed until the day it is re-armed.
        for key in ("skill_name", "skill_path"):
            value = _clean(fm.get(key))
            if value and not _is_nullish(value) and not skill_resolves(value, skills_dirs):
                where = " or ".join(str(d) for d in skills_dirs)
                if "/" in value or value.endswith(".md"):
                    warnings.append(f"{p.name}: {key} '{value}' is not a file that exists")
                else:
                    warnings.append(
                        f"{p.name}: {key} '{value}' resolves to no SKILL.md under {where}"
                    )
        model = _clean(fm.get("model"))
        if model and not _is_nullish(model):
            if known_models is _UNSET:
                known_models = model_names(config_path)
            if known_models is None:
                warnings.append(
                    f"{p.name}: model '{model}' unchecked — {config_path} has no "
                    "readable models: mapping"
                )
            elif model not in known_models:
                warnings.append(
                    f"{p.name}: model '{model}' is not a models: key or alias in "
                    f"{config_path} (known: {', '.join(sorted(known_models))})"
                )
        frequency = _clean(fm.get("frequency")).lower()
        rpd = _clean(fm.get("runs_per_day"))
        if frequency and not _is_nullish(frequency) and (not rpd or _is_nullish(rpd)):
            # `runs_per_day` is read first by the scheduler, so a task carrying
            # it dispatches whatever `frequency` says; without it the string
            # must be one the map knows or the task is parked with no signal.
            if FREQUENCY_INTERVALS is None:
                warnings.append(
                    f"{p.name}: frequency '{frequency}' unchecked — the scheduler's "
                    "FREQUENCY_INTERVALS could not be imported"
                )
            elif frequency not in FREQUENCY_INTERVALS:
                warnings.append(
                    f"{p.name}: frequency '{frequency}' is not one of "
                    f"{', '.join(sorted(FREQUENCY_INTERVALS))} and there is no "
                    "runs_per_day — the scheduler resolves no interval and will "
                    "never dispatch it"
                )
        # A declared `acceptance:` block (#623) is graded on every run; one that
        # cannot grade what it says (a regex that does not compile reads as a
        # false completion forever) is caught here, before the first run.
        if acceptance_problems is None and fm.get("acceptance"):
            warnings.append(f"{p.name}: acceptance unchecked — app.run_acceptance "
                            "could not be imported")
        elif acceptance_problems is not None:
            warnings.extend(f"{p.name}: {msg}" for msg in acceptance_problems(fm))
        dep = fm.get("depends_on")
        dep_id = _clean(dep)
        if dep and not _is_nullish(dep_id):
            if dep_id not in parsed:
                warnings.append(
                    f"{p.name}: depends_on '{dep}' not found among parseable tasks"
                )
            else:
                dep_status = _clean(parsed[dep_id].get("status"))
                if dep_status not in DEPENDABLE_STATUSES:
                    warnings.append(
                        f"{p.name}: depends_on {dep_id} but #{dep_id} is "
                        f"status={dep_status or 'unset'}"
                    )

    print(f"Scanned {len(files)} task files in {d}")
    print(f"  parseable:   {len(by_file)}")
    print(f"  UNPARSEABLE: {len(unparseable)}")

    if unparseable:
        print("\n🔴 CRITICAL — these tasks are INVISIBLE to the scheduler:")
        for name, err in unparseable:
            print(f"   {name}\n       {err}")

    if warnings:
        print(f"\n⚠️  {len(warnings)} structural warning(s):")
        for w in warnings:
            print(f"   {w}")

    if unparseable:
        return 1
    if warnings and args.strict:
        return 2
    print("\n✅ all task files parse" + (" and pass structural checks" if not warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
