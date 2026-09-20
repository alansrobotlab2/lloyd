#!/usr/bin/env python3
"""validate_skill_patch_queue.py — fail-loud checker for the skill-patch queue.

The queue at `~/lloyd/_pipeline/skills/proposed/` has no code writer and no code
reader: both are agents following prose (`nightly-skill-consolidation` Phase 3.1
writes entries, `nightly-skills-management` Stage 6 selects them), and before
this script nothing under `~/lloyd` parsed the directory at all — `grep -rn
"skills/proposed" --include=*.py` returned 0 hits when #1291 was triaged.

That matters because Stage 6 selects on the *parsed* `applied:` key. A file whose
front matter no parser accepts is therefore not rejected but skipped, silently:
from 2026-09-08 to 2026-09-20 four of the seven queue files were unparseable from
the day each was written (`generated: "2026-09-08T04:40:00Z",  # ← applied:
false` — the comma before the `#` makes YAML expect a flow sequence), and every
pass in between reported selecting on a parsed key. One of those files,
`patch-file-mutation-safety-2026-09-11.md`, carried `landed: false` plus a
`blocked_on:` line telling the next run to re-apply content that had been in
vault HEAD since `d0692ce4`. The 2026-09-19 pass appended a second
`held_at:`/`held_reason:` pair to two other files, which is a front matter no
strict loader accepts and a lenient one resolves arbitrarily.

Three defect classes, each one a file whose `applied:` cannot be trusted:

  * front matter that does not parse — or no front-matter block at all;
  * a mapping that repeats a key. `yaml.safe_load` takes the last value and
    raises nothing, so this needs its own loader
    (`duplicate_key_errors()`), and that is why the checker does not simply
    `safe_load` and catch;
  * a mapping that parses cleanly but has no `applied:` key — Stage 6 treats it
    as `false` and writes the key, so the checker names it rather than passing
    the file silently.

Exit codes:
    0  every queue file parses, has no duplicate key, and carries `applied:`
    1  at least one file has a defect — each offending file is named with its cause
    2  the queue directory is missing or unreadable, so no verdict is possible

Unlike `scripts/skill_lint.py` (:13, "Exit 0 always"), which is advisory and
scans only `~/obsidian/skills/*/SKILL.md`, this checker's whole purpose is the
non-zero exit: a failing queue file must stop the stage that is about to select
on it. `scripts/autonomy/validate_tasks.py` is the closer precedent.

Usage:
    ~/lloyd/.venvs/lloyd/bin/python ~/lloyd/scripts/validate_skill_patch_queue.py \
        [--dir ~/lloyd/_pipeline/skills/proposed]
"""
from __future__ import annotations

import argparse
import collections.abc
from pathlib import Path

import yaml

# Repo-relative default: scripts/ sits one level under the checkout root, and
# the queue is `_pipeline/skills/proposed/` in that same root. `_pipeline/` is
# gitignored, which is why the tests run against fixture dirs instead of here.
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "_pipeline" / "skills" / "proposed"

APPLIED_KEY = "applied"


def front_matter(text: str) -> str | None:
    """The leading `---` … `---` block's contents, or None if there is no block.

    Same convention the consumers use: the front matter is the file's first
    thing, delimited by a fence line at the top and the next `---` line.
    """
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    return text[3:end]


def duplicate_key_errors(fm: str) -> list[str]:
    """Duplicate keys in a front-matter mapping, found without trusting the loader's merge.

    PyYAML's constructor for `tag:yaml.org,2002:map` builds a dict, so a mapping
    that repeats a key resolves last-wins and raises nothing: safe_load on
    `applied: false\\nheld_at: a\\nheld_at: b` answers `{'applied': False,
    'held_at': 'b'}`. Walking `node.value` and counting keys is the only way to
    see the repetition, and the runbook's "write exactly one held-decision set"
    rule (`nightly-skills-management` Stage 6 step 4) is unenforceable without it.
    """
    errors: list[str] = []

    class _Loader(yaml.SafeLoader):
        """SafeLoader whose mapping constructor records repeated keys before merging them."""

    def _construct_mapping(loader, node, deep=False):
        mapping = loader.construct_mapping(node, deep=deep)  # last-wins, exactly as safe_load
        first_seen: dict[object, int] = {}
        for key_node, _value_node in node.value:
            try:
                key = loader.construct_object(key_node, deep=True)
            except yaml.YAMLError:
                continue
            if not isinstance(key, collections.abc.Hashable):
                continue
            line = key_node.start_mark.line + 1  # 1-based, within the front matter
            if key in first_seen:
                errors.append(
                    f"duplicate key {key!r} (front-matter line {first_seen[key]} "
                    f"repeated at line {line})"
                )
            else:
                first_seen[key] = line
        return mapping

    _Loader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
    )
    yaml.load(fm, Loader=_Loader)  # raises what safe_load raises; duplicates are collected
    return errors


def parse_failure_reason(exc: Exception) -> str:
    """The parser's own problem descriptions, not its echoed source lines.

    PyYAML's `str()` interleaves two kinds of line — the problem ("expected
    <block end>, but found ','") and an echo of the offending source indented by
    four spaces, plus `in "<unicode string>", line N, column M`. Only the
    unindented lines describe the defect, and for the comma-before-`#` shape the
    second one is the whole diagnosis.
    """
    problems = [
        line for line in str(exc).splitlines()
        if line and not line.startswith((" ", "\t")) and not line.startswith('in "')
    ]
    reason = "; ".join(problems[:2]) or type(exc).__name__
    mark = getattr(exc, "problem_mark", None)
    if mark is not None:
        reason += f" (front-matter line {mark.line + 1})"
    return reason


def check_file(path: Path) -> list[str]:
    """Every defect found in one queue file's front matter; empty means it is usable."""
    text = path.read_text(encoding="utf-8", errors="replace")
    fm = front_matter(text)
    if fm is None:
        return ["no front-matter block (a leading `---` fence and its closing `---`)"]

    try:
        # First the parse a consumer would do, so a parse error is reported with
        # the parser's own message rather than a downstream symptom.
        data = yaml.safe_load(fm)
    except Exception as exc:  # ScannerError, ParserError, ConstructorError, unhashable keys
        return [f"front matter does not parse: {parse_failure_reason(exc)}"]

    defects = duplicate_key_errors(fm)

    if data is None:
        defects.append("front matter is empty")
    elif not isinstance(data, dict):
        defects.append(f"front matter parses to {type(data).__name__}, not a mapping")
    elif APPLIED_KEY not in data:
        defects.append(f"no `{APPLIED_KEY}:` key (Stage 6 selects on the parsed key)")

    return defects


def check_queue(queue_dir: Path) -> list[tuple[Path, list[str]]]:
    """(file, defects) for each defective `*.md` in the queue, in name order."""
    out: list[tuple[Path, list[str]]] = []
    for path in sorted(queue_dir.glob("*.md")):
        defects = check_file(path)
        if defects:
            out.append((path, defects))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail loudly on an unparseable / duplicate-key / applied-less "
                    "skill-patch queue file."
    )
    parser.add_argument(
        "--dir", default=str(DEFAULT_DIR), type=Path,
        help=f"queue directory to validate (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args(argv)

    queue_dir: Path = args.dir.expanduser()
    if not queue_dir.is_dir():
        print(f"QUEUE DIR MISSING: {queue_dir} — no verdict is possible")
        return 2

    files = sorted(queue_dir.glob("*.md"))
    bad = check_queue(queue_dir)

    print(f"skill-patch-queue check: {queue_dir}")
    print(f"  files: {len(files)}")
    for path, defects in bad:
        for defect in defects:
            print(f"DEFECT {path}: {defect}")
    print(f"  defects: {len(bad)}")

    if bad:
        print(f"\n🔴 {len(bad)} of {len(files)} queue files cannot be trusted for "
              f"`{APPLIED_KEY}:` selection — fix them before Stage 6 selects.")
        return 1
    if not files:
        print("\n✅ queue is empty (no *.md to validate)")
    else:
        print(f"\n✅ all {len(files)} queue files parse, have no duplicate key, "
              f"and carry `{APPLIED_KEY}:`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
