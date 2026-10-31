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
  * `frequency:` and `runs_per_day:` contradicting each other (#2445) — #30
    declared `frequency: daily` beside `runs_per_day: 3` and the scheduler obeys
    the number, so a job that said once a day dispatched three times a day for 47
    days while this script reported 0 warnings over 38 files. The `frequency:`
    word being outside the map with NO `runs_per_day` beside it (#815) is the
    other half: that pair resolves no interval at all and never dispatches.
  * membership: whether every live task file is named by BOTH tables of
    `architecture/autonomy-jobs.md` (`DOC_MEMBERSHIP`, #2895), and — since #3270 — whether the
    drift journal those tables' filing check appends to still holds a row whose id no table
    names (`DRIFT_UNCLOSED`, which carries that row's date and so says how long the doc has been
    behind). Both are warnings, never exit rules short of `--strict`.
  * `requires_slot:` naming a program whose enabled flag is false (#1577) —
    `agent-llm-secondary` while `secondary_enabled: false`, the shape that retired
    #85. No name check can see this one: `secondary` is still a key of `models:`,
    so `model: primary` and `model: secondary` both resolve clean against a slot
    that was retired on 2026-09-20 (`551e9044`). Since #1555 the two status writers
    also refuse to *arm* such a task (`app/autonomy.py::slot_arm_block`); this is
    the static half, and it answers from `app.llm_slots`, never a copy.

Usage:
    python scripts/autonomy/validate_tasks.py [--autonomy-dir DIR] [--strict]
        [--skills-dir DIR ...] [--config FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
# Run by path (`python …/scripts/autonomy/validate_tasks.py`), which puts this file's own directory on
# `sys.path[0]` and not the checkout: without the insertion below, `app.data_root`
# raises `ModuleNotFoundError` and the reader silently never runs. Same reason as
# `scripts/skill_lint.py`'s insertion. (#3403)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from app.data_root import vault_root  # noqa: E402

# The scheduler's own frequency vocabulary (#815): a `frequency:` outside it
# with no `runs_per_day` resolves to no interval and never dispatches, silently
# until 2026-09-24. Imported rather than restated so the linter cannot drift
# from the map it lints against. `None` means the check could not run.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from app.autonomy import FREQUENCY_INTERVALS  # noqa: E402
except Exception:  # noqa: BLE001 — a linter with no scheduler beside it
    FREQUENCY_INTERVALS = None
# The grader's own reading of an `acceptance:` block (#623), for the same reason.
try:
    from app.run_acceptance import acceptance_problems  # noqa: E402
except Exception:  # noqa: BLE001
    acceptance_problems = None
# The one authority on "is this program supposed to be running" (#1577). Imported
# for the same reason as FREQUENCY_INTERVALS: a second slot table in the linter is
# a second thing to drift when a slot is retired, and `app.llm_slots`'s own
# docstring is the record of what six tables that silently disagree cost. The
# module is cheap to import — it reaches for the live app.config only when a
# caller hands it no config at all, and the linter always hands it the one read
# from `--config`. None = the check cannot run, which is reported, never passed.
try:
    from app import llm_slots  # noqa: E402
except Exception:  # noqa: BLE001
    llm_slots = None
# The membership tables' parser (#2895), which lives beside this file rather than in
# `tests/` precisely so the linter, the two task-file writers and the nightly reconciler
# share ONE reading of the two tables. A second copy of the rule is how the class of drift
# this file now reports survived four rounds of tests.
#
# Imported under its package name, which is also how BOTH writers import it
# (`app/routers/autonomy.py:320`, `agent_mcp/autonomy.py:407`) — and that is load-bearing, not
# tidiness. `import jobs_membership` off a sys.path entry naming this directory builds a SECOND
# module object for the same file, and `jobs_membership.journal_path` is a value inside one of
# them: redirecting the store by patching the module (the seam #3270's reader tests use, and the
# one `tests/test_autonomy_task_membership_at_filing.py:373` already patches) would then aim at
# the writers' copy and leave this one reading the real journal. One name, one object, one patch.
try:
    from scripts.autonomy import jobs_membership as jm  # noqa: E402
except Exception:  # noqa: BLE001
    jm = None
# Which supervisord program each `models:` alias tracks (#2632). `requires_slot:`
# names a program, so #1577's check could ask `llm_slots` directly; `model:` names an
# ALIAS, and the one place the tree says which row an alias belongs to is
# `supervisor_client._MODEL_ROW_ALIASES`. Inverting it here rather than writing
# `{"secondary": "agent-llm-secondary"}` into this file is the same rule as above: the
# linter must not own a table that can outlive the engine it describes. Inverted at
# import because the check asks alias-first. None = no verdict, which is reported.
try:
    from app.supervisor_client import _MODEL_ROW_ALIASES as _ROW_ALIASES  # noqa: E402

    MODEL_ROW_ALIASES: dict[str, str] | None = {
        alias: program for program, alias in _ROW_ALIASES.items()
    }
except Exception:  # noqa: BLE001
    MODEL_ROW_ALIASES = None

DEFAULT_DIR = vault_root() / "autonomy"
# Mirrors the slug branch of app.autonomy._load_skill_content, which is the
# only skills root that path consults. --skills-dir overrides it.
DEFAULT_SKILLS_DIR = vault_root() / "skills"
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


#: How far a `runs_per_day:` may sit from the `frequency:` word written beside it
#: before the pair is a contradiction instead of a rounding (#2445). Both ends of
#: the gap are measured on the live board, not guessed: the three files carrying
#: `weekly` + `runs_per_day: 0.14` resolve 617142.9 s against the word's 604800 s,
#: 2.04% apart because 0.14 is two decimals of 1/7, while #30's `daily` + `3`
#: resolved 28800 s against 86400 s — 66.67% apart, which is the defect that fired
#: a daily job three times a day. It stays a warning, never a parse failure: a task
#: that genuinely means 0.15 runs/day still dispatches on its number.
_RPD_AGREEMENT_TOLERANCE = 0.05


def _runs_per_day_seconds(value) -> float | None:
    """What a `runs_per_day:` resolves to, or None when it resolves to nothing.

    Mirrors the scheduler's own read of the field (`app/autonomy.py:1227-1238`):
    `86400 / float(rpd)`, reached only for a positive number, with a non-number or
    a zero falling through to the `frequency:` word. Mirroring the fallback is what
    keeps this check quiet on a value the scheduler would have ignored anyway.
    """
    text = _clean(value)
    if _is_nullish(text):
        return None
    try:
        runs = float(text)
    except ValueError:
        return None
    return 86400.0 / runs if runs > 0 else None


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


def config_mapping(config_path: Path) -> dict | None:
    """The `--config` yaml as a mapping, or None when there is no readable mapping.

    Every check that reads it treats None as "no verdict" rather than "clean", and
    for `requires_slot` that distinction is load-bearing: `llm_slots.is_enabled`
    answers False for a slot whose flag key is simply *absent*, so a config that
    failed to parse would otherwise print a switched-off verdict that nobody
    configured.
    """
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — an unreadable config is the case being handled
        return None
    return cfg if isinstance(cfg, dict) else None


def slot_problem(value: str, config: dict | None, config_path: Path) -> str | None:
    """Why a `requires_slot:` value cannot be armed, or None when it can.

    `requires_slot` names the supervisord program the task's engine runs on, and
    two of them are optional with a switch in config.yaml. The answer is
    `app.llm_slots`'s — the same `is_enabled` the dispatch-side arming guard
    consults since #1555 — because a linter that kept its own list of slots would
    be the seventh table that disagrees about a retired engine.

    None means nothing to say: a program that is not an optional slot has no
    switch and is always enabled (`agent-llm-primary`), and a slot whose flag is
    true is what a task is supposed to name. The reported case is #85's:
    `requires_slot: agent-llm-secondary` held for a week against
    `secondary_enabled: false` while every name check stayed silent, because
    `secondary` remains a key of `models:` whatever the flag says.
    """
    if llm_slots is None:
        return (f"requires_slot '{value}' unchecked — app.llm_slots could not be "
                "imported, so no slot can be reported as enabled")
    flag = llm_slots.slot_flag(value)
    if flag is None:
        return None                     # not an optional slot: always enabled
    if config is None:
        return (f"requires_slot '{value}' unchecked — {config_path} holds no "
                "readable yaml mapping, so the slot's enabled flag could not be read")
    if llm_slots.is_enabled(value, config):
        return None
    return (f"requires_slot '{value}' names a switched-off slot: {flag} is false in "
            f"{config_path}, so nothing may arm this task and its engine is not "
            "booted to run it")


def model_slot_problem(value: str, config: dict | None, config_path: Path) -> str | None:
    """Why a `model:` value names a slot the box has switched off, or None when it does not.

    The sibling of `slot_problem`, for the field that is actually populated. That one
    reads `requires_slot:`, which NO task file declared as of 2026-10-12
    (`grep -ln requires_slot ~/obsidian/autonomy/*.md` → nothing); this one reads
    `model:`, which every task file that names an engine has. #68
    `morning-brief-triage` declared `model: secondary` from 2026-04-22 while
    `secondary_enabled: false` has been in force since 2026-09-20 (`551e9044`), and
    both runtime guards skip it for the same reason: `slot_arm_block`
    (app/autonomy.py:5706-5710) returns None for a task with no `requires_slot:`, and
    the name check below cannot see a flag — `secondary` stays a key of `models:`
    whatever the switch says.

    None means nothing to say, in three different ways worth telling apart: an alias
    no engine row tracks has no switch to be off; `agent-llm-primary` is not an
    optional slot, so `llm_slots` reports it always enabled whatever the flags say;
    and a slot whose flag is true is exactly what a task should name. The reported case
    is not an arming refusal — arming #68 succeeds today, and
    tests/test_autonomy_slot_arm_guard.py:113 pins that — it is that every run lands on
    the primary through `resolve_model_alias` (app/config.py:500-507) while the task
    file, and anyone reading it, believes otherwise.
    """
    if MODEL_ROW_ALIASES is None:
        return (f"model '{value}' unchecked — app.supervisor_client's alias↔row map "
                "could not be imported, so no model could be traced to a slot")
    program = MODEL_ROW_ALIASES.get(value)
    if program is None:
        return None                 # no engine row tracks this alias: no switch to be off
    if llm_slots is None:
        return (f"model '{value}' unchecked — app.llm_slots could not be imported, so "
                "no slot can be reported as enabled")
    flag = llm_slots.slot_flag(program)
    if flag is None:
        return None                 # not an optional slot: always enabled
    if config is None:
        return (f"model '{value}' unchecked — {config_path} holds no readable yaml "
                "mapping, so the slot's enabled flag could not be read")
    if llm_slots.is_enabled(program, config):
        return None
    return (f"model '{value}' names a switched-off slot: {flag} is false in "
            f"{config_path}, so {program} is not booted and `resolve_model_alias` "
            f"routes every run of this task to the primary — the file says {value!r} "
            "and the box serves primary; set model: to the engine it should actually "
            "run on, or boot the slot")


def model_names(config_path: Path) -> set[str] | None:
    """Every `model:` value the engine will accept: keys of `models:` + aliases.

    Mirrors autonomy._get_model_env (:1141-1155): it matches the name as a key of
    `models:`, then as any entry's `alias`, then returns {}. So a value outside
    this set is exactly a value that reaches the endpoint unrecognized.

    Returns None when no `models:` mapping can be read — the caller must treat
    that as "cannot check", never as "nothing to warn about".
    """
    config = config_mapping(config_path) or {}
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
    """Whether app.autonomy._load_skill_content would return content for this value.

    A value containing '/' or ending in .md is used as a path and nothing else —
    _load_skill_content's path branch returns None on a miss rather than retrying
    it as a slug. Anything else is a slug, resolved by that loader's slug branch
    against each skills dir as `<dir>/<slug>/SKILL.md`. A path naming a directory
    fails the same way dispatch does: read_text raises, loader returns None.
    """
    expanded = Path(value.replace("~", str(Path.home())))
    if "/" in value or value.endswith(".md"):
        return expanded.is_file()
    for d in skills_dirs:
        if (d / expanded / "SKILL.md").is_file():
            return True
    return False


#: The drift journal's default location, for the one run in which `jobs_membership` did not
#: import. `jm.journal_path()` is the authority — it is what the writer appends through and what
#: a test redirects — and this is the same pipeline-relative path spelled out so the journal
#: reader below reports on a store instead of going silent when the import above failed. The
#: node at `tests/test_autonomy_task_membership_at_filing.py:250` pins what that function
#: resolves to, which is what keeps this fallback from drifting away from it.
JOURNAL_PATH = (Path.home() / "lloyd-data" / "_pipeline" / "arch-review"
                / "membership-drift.jsonl")


def drift_closed_since(doc_text: str, row: dict, parser) -> bool:
    """Is a journal row's drift closed — does every table it says was missing name it now?

    A named predicate rather than a line inside the loop below, because this is the exact question
    the clause under test asks, and because it is the only question the journal can be asked:
    #2813, #2881 and #3012 each end with somebody hand-adding the rows, so "is it STILL open" is
    what decides whether a row is history or news.

    `parser` is the caller's `jobs_membership`, which may be None; a row whose tables cannot be
    re-read counts as STILL OPEN rather than closed. The reason is the shape of the failure: an
    unreadable doc is the one state in which every row would look closed, and a reader that goes
    quiet exactly when the doc breaks is the reader that has already failed.
    """
    tid = str(row.get("task_id") or "").strip()
    tables = [str(t) for t in (row.get("tables_missing") or [])]
    if not tid or not tables:
        return True                                   # not a row this class can speak about
    if parser is None:
        return False
    try:
        return all(tid in parser.named_ids(doc_text, header)
                   for name, header in parser.TABLES if name in tables)
    except parser.UnreadableDoc:
        return False


def drift_journal_warnings(jobs_doc: Path, journal: Path | None = None,
                           parser=None) -> list[str]:
    """DRIFT_UNCLOSED warnings — the journal rows whose drift the doc has not taken back.

    `jobs_membership.report_at_filing` appends one row the moment a CREATED task file's id is
    missing from a membership table, carrying the `ts` of that filing. Nothing read the file: at
    the HEAD this was written on, `git grep -l membership-drift` named the writer and two test
    files and no script, and the drift reached a human three times (#2813, #2881, #3012) only
    through the `@live_vault` nodes the automod gate deselects
    (`scripts/automod/gate.py:719`, `TESTS_MARK_EXPR`). So the reader goes where a nightly looks
    and never under `scripts/automod/`: the bound at
    `tests/test_autonomy_task_membership_at_filing.py:303-305` is that no promotion or rollback
    path reads this journal, because documentation must not gate execution.

    Every line carries the row's date, because `ts` is the journal's ONLY unique content.
    `doc_membership_warnings` above already prints today's drift from the tables and the
    directory on every run, so a line that only repeated it would be a second copy of a check
    this file already runs; what only the journal knows is WHEN the drift started, which is what
    tells "the doc is a day behind" from "behind since the file landed, and nobody saw it".

    An absent journal prints nothing and reports nothing. That is the normal state, not a fault:
    task creates run about twice a week, the doc has been kept since #2881, and the store has
    never held a row on this machine — a class that reads red on an empty store is muted inside a
    day, which is the failure this item's own traffic report warned against. Only a store that
    HOLDS rows and cannot be read or parsed says `unchecked`.

    Which store is meant follows `app.paths.PIPELINE_DIR`, and that is anchored to the tree the
    import came from: run from an automod worktree, this reads the worktree's own empty
    `.lloyd-data/_pipeline/`, and run from the live checkout it reads
    `~/lloyd-data/_pipeline/`. So the readings a gate run prints are about the live fleet only
    when the nightly runs it from the live tree — name the tree, as `app.paths.describe_tree()`
    does, before calling a silent report a clean one.
    """
    jm = parser if parser is not None else globals().get("jm")
    path = Path(journal) if journal else (jm.journal_path() if jm else JOURNAL_PATH)
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"DRIFT_UNCLOSED: unchecked — {path} could not be read: "
                f"{type(exc).__name__}: {exc}"]
    try:
        text = Path(jobs_doc).read_text(encoding="utf-8")
        doc_error = ""
    except OSError as exc:
        text = ""
        doc_error = f"{jobs_doc} unreadable: {type(exc).__name__}: {exc}"

    out: list[str] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            out.append(f"DRIFT_UNCLOSED: unchecked — {path}:{lineno} is not a json object: {exc}")
            continue
        if not isinstance(row, dict):
            out.append(f"DRIFT_UNCLOSED: unchecked — {path}:{lineno} is not a json object")
            continue
        if drift_closed_since(text, row, jm):
            continue
        tid = str(row.get("task_id") or "?")
        if doc_error:
            out.append(f"{tid}: DRIFT_UNCLOSED: unchecked — open since {row.get('ts')}; "
                       f"{doc_error}, so its tables could not be re-read")
            continue
        missing = ", ".join(str(t) for t in (row.get("tables_missing") or []))
        out.append(f"{tid}: DRIFT_UNCLOSED: open since {row.get('ts')} — {path.name} says {missing} "
                   f"omitted it when the file was filed, and it still does "
                   f"(doc {row.get('doc_short_sha')})")
    return out


def doc_membership_warnings(d: Path, jobs_doc: Path,
                            parser=None) -> list[str]:
    """DOC_MEMBERSHIP and DRIFT_UNCLOSED warnings for the fleet in `d` against `jobs_doc`.

    Separate from `main()` so `tests/test_validate_tasks_linter.py` can assert on the
    messages over a fixture pair without running the scheduler's whole field validation over
    a synthetic file, and so the rule lives in one place with the parser it uses.

    The ids come from the DIRECTORY and the tables come from the DOC, both parsed by
    `scripts/autonomy/jobs_membership.py` — never from a list of expected ids in this file.
    A roster here would be the fifth hand-typed membership table in the tree, which is the
    thing that keeps drifting.

    The drift journal rides along here rather than beside here, because the two say one thing:
    the directory-and-tables scan is what decides whether a journal row is still open, so asking
    it twice in two places is how a report gets its two opinions. It is also the only seam that
    reads the doc once — the journal's rows are graded against the same text this function
    already parsed, and both classes share the scoping rule that a doc the caller never named is
    not this run's business (`main`'s `--jobs-doc` gate).
    """
    jm = parser if parser is not None else globals().get("jm")
    if jm is None:
        return [f"DOC_MEMBERSHIP: unchecked — jobs_membership.py is not importable from "
                f"{Path(__file__).parent}",
                *drift_journal_warnings(jobs_doc, parser=None)]
    try:
        text = Path(jobs_doc).read_text(encoding="utf-8")
        live = jm.task_file_ids(Path(d))
        retired = jm.retired_named_ids(text)
        parsed = [(name, jm.named_ids(text, header)) for name, header in jm.TABLES]
    except (OSError, jm.UnreadableDoc) as exc:
        # A doc whose tables cannot be located is not a doc in sync. Reported, never
        # silently passed: `table_rows` raises rather than parsing nothing, because an empty
        # parse and a clean doc would otherwise look identical from here. Both tables are read
        # inside this `try` — the header is looked up per table, so a loop that parsed outside
        # it raised straight out of the linter on a doc with its headings reworded, which is
        # exactly the shape the exception exists for.
        #
        # The journal is still read on this path: a row's date is a fact about the store and does
        # not depend on the doc parsing, and `drift_journal_warnings` reports its own rows as
        # `unchecked` rather than closed when the tables cannot be re-read.
        return [f"DOC_MEMBERSHIP: unchecked — {jobs_doc}: {exc}",
                *drift_journal_warnings(jobs_doc, parser=parser)]
    out: list[str] = []
    for name, named in parsed:
        ghosts, unlisted, dupes = jm.table_drift(named, live, retired)
        out += [f"{i}: DOC_MEMBERSHIP: {name} names no row for a live task file — add its "
                f"row or retire it in {Path(jobs_doc).name}" for i in unlisted]
        out += [f"{i}: DOC_MEMBERSHIP: {name} names a job with no task file under {d} and "
                f"no Retired row" for i in ghosts]
        out += [f"{i}: DOC_MEMBERSHIP: listed more than once in {name}" for i in dupes]
    out += drift_journal_warnings(jobs_doc, parser=parser)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the linter and return its exit code. `argv=None` is the real command line.

    The parameter exists because of one class: `DRIFT_UNCLOSED` reads a store whose path
    `jobs_membership.journal_path()` resolves INSIDE this process, so a test that seeds a
    journal has to run the check in the process holding the patch, and cannot go out through a
    subprocess. Nothing else about the entry point changes — `python scripts/autonomy/
    validate_tasks.py …` still calls this with nothing and parses `sys.argv`, the flags are the
    same ones, and the codes are the documented ones.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--autonomy-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--skills-dir", action="append", default=None,
                    metavar="DIR",
                    help="directory of <slug>/SKILL.md to resolve skill_name "
                         "against; repeatable. Default: ~/obsidian/skills, the "
                         "only skills root the dispatch loader consults.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="yaml whose `models:` keys and aliases define the "
                         "accepted model: values, and whose slot enabled flags "
                         "(`secondary_enabled`, `djev.enabled`) decide which "
                         "requires_slot: programs are switched off")
    ap.add_argument("--jobs-doc", default=None, metavar="FILE",
                    help="architecture/autonomy-jobs.md whose two membership tables are "
                         "checked against the directory (DOC_MEMBERSHIP), and whose same "
                         "tables decide whether the drift journal's rows are still open "
                         "(DRIFT_UNCLOSED). Default: that doc in the tree this script runs "
                         "from, but only when --autonomy-dir is the default fleet directory; "
                         "pass it explicitly to check a fixture or a redirected fleet. Both "
                         "classes are skipped without it — a store this run was never asked "
                         "about is not its business.")
    ap.add_argument("--strict", action="store_true",
                    help="treat structural warnings as failures (exit 2)")
    args = ap.parse_args(argv)

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
    # Same for the slot enabled flags: read once, only if some file declares a
    # requires_slot OR names a model (#2632 — both legs ask the same question of the
    # same file), and an unreadable mapping is reported per file — `is_enabled`
    # answers False for a flag that is absent, so silence here would be a verdict.
    slot_cfg = _UNSET

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
        # at dispatch with "Skill not found" (app/autonomy.py:1435). Checked for every
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
            # The name check above answers "does an engine serve this name"; only the
            # flag answers "is that engine booted". Both run, because they fail
            # differently: `model: eco` reaches an endpoint that does not serve it,
            # `model: secondary` reaches the PRIMARY's endpoint under a retired slot's
            # name and every guard waves it through (#2632). Shares `slot_cfg` with the
            # `requires_slot:` leg below — one read of the same file, one None meaning.
            if slot_cfg is _UNSET:
                slot_cfg = config_mapping(config_path)
            # isinstance, not `is _UNSET`: the sentinel is an object(), and the
            # ternary's type is what tells a reader (and pyright) that a missing
            # config reaches the check as None rather than as something dict-ish.
            model_cfg: dict | None = slot_cfg if isinstance(slot_cfg, dict) else None
            problem = model_slot_problem(model, model_cfg, config_path)
            if problem:
                warnings.append(f"{p.name}: {problem}")
        slot = _clean(fm.get("requires_slot"))
        if slot and not _is_nullish(slot):
            # A name check cannot see this: `secondary` is a models: key whether or
            # not the slot is booted, so #85 passed every check above for a week
            # against `secondary_enabled: false` (#1577).
            if slot_cfg is _UNSET:
                slot_cfg = config_mapping(config_path)
            problem = slot_problem(slot, None if slot_cfg is _UNSET else slot_cfg,
                                   config_path)
            if problem:
                warnings.append(f"{p.name}: {problem}")
        frequency = _clean(fm.get("frequency")).lower()
        rpd = _clean(fm.get("runs_per_day"))
        rpd_seconds = _runs_per_day_seconds(fm.get("runs_per_day"))
        if frequency and not _is_nullish(frequency):
            # `runs_per_day` is read FIRST by the scheduler, so a task carrying a
            # positive numeric one dispatches at 86400/n whatever its `frequency:`
            # word says; the word only rules when the number is absent, nullish,
            # zero or unparseable. Two ways to get that wrong, and until #2445 the
            # gate below only ever looked at the first one, because the whole
            # block sat behind `(not rpd or _is_nullish(rpd))` — a file carrying
            # BOTH fields was never compared with itself.
            if FREQUENCY_INTERVALS is None:
                if rpd_seconds is None:
                    # Nothing resolves an interval, and the map needed to say so
                    # could not be imported.
                    warnings.append(
                        f"{p.name}: frequency '{frequency}' unchecked — the scheduler's "
                        "FREQUENCY_INTERVALS could not be imported"
                    )
            elif frequency not in FREQUENCY_INTERVALS:
                # Out of the four-word map is legal only when the number carries the
                # schedule: #24's `6x-daily` + `runs_per_day: 6` is that idiom.
                if rpd_seconds is None:
                    warnings.append(
                        f"{p.name}: frequency '{frequency}' is not one of "
                        f"{', '.join(sorted(FREQUENCY_INTERVALS))} and there is no "
                        "runs_per_day — the scheduler resolves no interval and will "
                        "never dispatch it"
                    )
            elif rpd_seconds is not None:
                # Both fields speak. The word is the human-declared period, so a
                # number that disagrees with it is the file contradicting itself —
                # and the number is the one that wins, which is why #30 ran 3x/day
                # for 47 days reporting `frequency: daily` the whole time.
                declared = FREQUENCY_INTERVALS[frequency]
                if abs(rpd_seconds - declared) > declared * _RPD_AGREEMENT_TOLERANCE:
                    warnings.append(
                        f"{p.name}: frequency '{frequency}' is {declared:g}s but "
                        f"runs_per_day {rpd} resolves {rpd_seconds:g}s — the "
                        "scheduler dispatches at the number, so runs_per_day wins "
                        "and this job does not run on the period it declares; make "
                        "the two agree, or take the label outside the four-word map "
                        "the way #24's `6x-daily` + `runs_per_day: 6` does"
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

    # DOC_MEMBERSHIP (#2895) — is every live task file named by BOTH membership tables of
    # `architecture/autonomy-jobs.md`? Ids come from the directory and both tables are
    # parsed, so there is no allowlist to age and no file to register: the file on disk IS
    # the fleet, which is the rule the four rounds on this class (#1102, #2630/#2631, #2813,
    # #2881) kept rediscovering from the wrong end, after a suite went red.
    #
    # Only this class joins `warnings`, never the exit rules below: the doc is prose, and
    # prose lagging the fleet by a day has never been a reason to fail a scheduler's
    # validation. `--strict` still turns it into exit 2, because that flag's whole meaning is
    # "treat structural warnings as failures".
    #
    # It runs when the doc was asked for (`--jobs-doc`) or when the directory is the default
    # one, and is skipped otherwise. A test passing `--autonomy-dir tmp` is asking about one
    # file's fields and has no opinion about the shared doc, and `jm.JOBS_DOC` resolves to
    # whichever tree is running, so grading a real fleet against a checkout that is behind it
    # would be a measurement of the worktree.
    jobs_doc = Path(args.jobs_doc).expanduser() if args.jobs_doc else None
    if jobs_doc is None and d.resolve() == DEFAULT_DIR.resolve():
        jobs_doc = jm.JOBS_DOC if jm else None
    if jobs_doc is not None:
        warnings.extend(doc_membership_warnings(d, jobs_doc, jm))

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
