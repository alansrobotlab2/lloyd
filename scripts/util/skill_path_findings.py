#!/usr/bin/env python3
"""Name the dead repo paths a staged skill or task edit ADDS, before commit.

Item #1969. Two suite nodes read the live vault and go red on `main` when a
skill names a checkout path that is not there:
`tests/test_skill_tool_names.py::test_no_active_skill_or_task_names_a_path_absent_from_the_checkout`
(unmarked, so it is in every round's `tests` rung) and
`tests/test_bench_audit_tasks.py::test_skill_dead_path_gold_still_matches_the_skills`
(`live_vault`). Both look only after the fact, and the writers are nightly jobs
that rewrite skill prose every night: `web/.vite` (#1886), then `~/lloyd/.t`
removed by hand on 2026-09-30 (`0473a7e2`) and re-added by the next knowledge
write six hours later (`cc10159d`), then two strays quoted in an incident note
(#2026, main red at `219e1314`). Each time the sentence was TRUE — "this path is
absent" — which is why reading it does not catch it: a dead-path check cannot tell
"this does not exist, as the text says" from "this is where the fix lives".

A vault-side rule in the writing skill would be rewritten by the same jobs, so the
check lives here, in tracked code, at the one door those jobs commit through:
`scripts/util/vault-commit.sh` runs this after staging and before committing.

This is a REPORT, not a gate, for the reason `autonomy_status_findings.py` gives
and one more: the wrapper's no-pathspec mode commits other writers' state, so a
refusal would let one job's dead path block every later job's pre-flight
snapshot. One line per NEWLY ADDED dead reference, exit 0, the commit proceeds;
the line says how to reword, and the job that reads its own commit output can
amend in the same run instead of leaving main red for a hand sweep.

Only what the staged edit adds is reported (rows in the index minus rows at
HEAD), so pre-existing drift — `PATH_KNOWN_UNFIXED`, the gold items of
`lloyd/bench/bench_016_audit_skill_dead_paths.md` — never prints: a permanent
alarm is a disabled alarm.

The path rule is NOT restated here. It is `_absent_refs`, `_tree_ignores` and
`PATH_KNOWN_UNFIXED` in `tests/test_skill_tool_names.py`, loaded from the same
checkout this script sits in, so the writer-side answer cannot drift from the
node's. That module imports `pytest`; under an interpreter without it this
prints CHECK SKIPPED and exits 0 (the wrapper prefers the repo venv, which has
it). The bench node's narrower rule is the one piece defined here —
`bench_dead_refs` — and that node calls it, for the same one-definition reason.
That leg subtracts the loaded guard's `_creation_sites` too: paths a page says its
own job writes, whether at a flag, a redirection or in an English sentence, are
outputs and not citations (#2157, #2223), and one commit's two legs must not print
different verdicts about one sentence.

The one edge the delete side gets a line for (#4469): a staged `skills/**/SKILL.md`
edit that STOPS naming a dead path whose row is still gold in
lloyd/bench/bench_016_audit_skill_dead_paths.md gets a single
`skill-path GOLD REFRESH:` line naming that card, because `gold_items` is a
hand-maintained answer key and the edit just orphaned one of its rows. Both
instances were silent commits: vault `07379a6ac` repointed `service-health-check`
off `~/lloyd/autonomy.db` (#4461) and vault `dd9530a1` deleted
`file-path-resolution`'s `~/lloyd/inner-voice/system_prompt.md` line (#3110), and
each left the symmetric node red for a day. It is a notice, not a finding: the
count line and the exit status are what they were, and `autonomy/*.md` is not its
corpus because the node walks `skills/**/SKILL.md` alone. A whole-file `SKILL.md`
deletion never reaches it either — `staged_docs` is `--diff-filter=AM` — which is
the wider silent edge #4469 deliberately leaves open.

    python3 ~/lloyd/scripts/util/skill_path_findings.py --repo ~/obsidian
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
# Run by path (`python …/scripts/util/skill_path_findings.py`), which puts this file's own directory on
# `sys.path[0]` and not the checkout: without the insertion below, `app.data_root`
# raises `ModuleNotFoundError` and the reader silently never runs. Same reason as
# `scripts/skill_lint.py`'s insertion. (#3403)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from app.data_root import vault_root  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GUARD = REPO_ROOT / "tests" / "test_skill_tool_names.py"

#: A staged path this rung reads: an active skill page or an autonomy task.
_DOC_RE = re.compile(r"^(?:skills/[^/]+/SKILL\.md|autonomy/[^/]+\.md)$")

#: bench_016's reference rule: a `~/lloyd/…` token whose last component has a
#: dot (a file, or a dot-directory like `.t`), absent from the code tree.
BENCH_REF = re.compile(r"(?:~|/home/alansrobotlab)/lloyd/([A-Za-z0-9_./-]*[A-Za-z0-9_-])")
#: The one path a skill presents AS a wrong example.
BENCH_WRONG_EXAMPLES = frozenset({"lloyd/inner-voice/system-prompt.md"})


def bench_dead_refs(text: str, lloyd_root: Path) -> set[str]:
    """Lower-cased `~/lloyd/`-relative paths `text` names that `lloyd_root` lacks."""
    dead: set[str] = set()
    for m in BENCH_REF.finditer(text):
        rel = m.group(1)
        if "." not in Path(rel).name or rel in BENCH_WRONG_EXAMPLES:
            continue  # directories and the presented-as-wrong example
        if not (lloyd_root / rel).exists():
            dead.add(rel.lower())
    return dead


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def _show(repo: Path, spec: str) -> str:
    """A blob's text, or "" when that side has no such file (added / deleted)."""
    try:
        return _git(repo, "show", spec)
    except RuntimeError:
        return ""


def staged_docs(repo: Path) -> list[str]:
    """Skill pages and task files added or modified in the index, sorted."""
    out = _git(repo, "diff", "--cached", "--name-only", "-z", "--diff-filter=AM",
               "--", "skills/", "autonomy/")
    return sorted(p for p in out.split("\0") if _DOC_RE.match(p))


def _load_guard():
    """The suite's own path rule, from this checkout. Raises if it cannot load."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_skill_path_guard", GUARD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def path_findings(repo: Path, guard=None) -> tuple[list[str], int]:
    """(findings, staged docs examined) for one repo's index."""
    guard = guard or _load_guard()
    # A `~/obsidian/…` reference is resolved against the repo being committed.
    # This module instance is private to this process, so the rebind reaches
    # nothing else.
    guard.VAULT = repo
    findings: list[str] = []
    docs = staged_docs(repo)
    for rel in docs:
        old, new = _show(repo, f"HEAD:{rel}"), _show(repo, f":{rel}")
        skill_dir = (repo / rel).parent if rel.startswith("skills/") else None
        # The node's recorded sites — what its subject writes (flag, redirection,
        # `mkdir` or prose sentence) and what a quoted exception says is missing
        # (#2223) — subtracted from the bench leg because that leg's path rule is
        # narrower than `_named_paths` and is defined in THIS file: the node has no
        # idea these lines exist, so the one place the two definitions can meet is
        # here. `_recorded_sites` is the same name `_absent_refs` consults, so a new
        # mention type reaches this leg by being added there, never by a second list
        # here. Without the subtraction a skill page recording that its job wrote a
        # dated report prints a finding the unmarked node does not raise — two
        # verdicts about one sentence, from one commit.
        created = {p.lower() for _tree, p in guard._recorded_sites(new)}
        added = (guard._absent_refs(rel, new, skill_dir)
                 - guard._absent_refs(rel, old, skill_dir)
                 - guard.PATH_KNOWN_UNFIXED)
        repo_rels = [guard._row_parts(r)[1] for r in added
                     if guard._row_parts(r)[0] == "repo"]
        ignored = guard._tree_ignores(guard.ROOT, repo_rels)
        named: set[str] = set()
        for row in sorted(added):
            tree, path = guard._row_parts(row)
            if tree == "repo" and path in ignored:
                continue
            named.add(path.lower())
            where = "the vault" if tree == "vault" else f"the checkout at {guard.ROOT}"
            findings.append(f"{rel}: newly names `{path}`, which is not in {where}")
        if rel.startswith("skills/"):
            for path in sorted(bench_dead_refs(new, guard.ROOT)
                               - bench_dead_refs(old, guard.ROOT) - named - created):
                findings.append(f"{rel}: newly names `~/lloyd/{path}`, which is not "
                                f"in the checkout at {guard.ROOT}")
    return findings, len(docs)


def _gold_keys(repo: Path) -> dict[str, set[str]]:
    """The answer keys the repo being committed carries, card by card.

    {card path relative to `repo`: lower-cased `gold_items` values}, read through
    `scripts.autoresearch.common.load_bench_tasks` — the loader the bench nodes use,
    so this cannot disagree with them about what a card or a gold item is. Cards with
    unparseable front matter are skipped by that loader, which is what a reporting
    rung wants: a broken card prints nothing instead of reaching a commit.

    From THIS repo and not from `vault_root()`, because `report()` is handed the repo
    under commit and a card set belonging to some other tree would be the wrong
    answer key for it. Values are compared as the lower-cased lloyd-relative rels
    `bench_dead_refs` returns, since that is how bench_016 stores them
    (`- scripts/memory/extract-session-log.py`, no `~/lloyd/` prefix).
    """
    from scripts.autoresearch.common import load_bench_tasks  # stdlib + yaml only

    keys: dict[str, set[str]] = {}
    for task in load_bench_tasks(repo / "lloyd" / "bench"):
        items: set[str] = set()
        for check in task.get("objective_checks") or []:
            if isinstance(check, dict):
                items |= {str(g).strip().lower()
                          for g in check.get("gold_items") or []}
        if items:
            keys[Path(task["_path"]).relative_to(repo).as_posix()] = items
    return keys


def gold_refresh(repo: Path, guard=None) -> dict[str, list[str]]:
    """Gold rows a staged skill edit orphans, keyed by the card still listing them.

    The delete edge `path_findings` deliberately does not compute: refs at HEAD minus
    refs in the index, over the corpus the gold node walks — `skills/**/SKILL.md`,
    `.archived` skipped — because an `autonomy/*.md` edit that drops a `~/lloyd/…`
    path can never orphan a gold item and printing one there would be a false alarm
    on the task-file corpus. The result is intersected with each card's gold rows, so
    a removed path no card lists stays silent.

    The removal is per edited doc rather than a re-walk of the whole corpus, which
    would mean reading every skill page in the vault at commit time: a path another
    active skill still names keeps the corpus intact, and the advice below carries
    that caveat instead of this loop guessing at it.
    """
    guard = guard or _load_guard()
    cards = _gold_keys(repo)
    if not cards:
        return {}
    removed: set[str] = set()
    for rel in staged_docs(repo):
        if not rel.startswith("skills/") or ".archived" in Path(rel).parts:
            continue
        removed |= (bench_dead_refs(_show(repo, f"HEAD:{rel}"), guard.ROOT)
                    - bench_dead_refs(_show(repo, f":{rel}"), guard.ROOT))
    out: dict[str, list[str]] = {}
    for card, gold in cards.items():
        orphaned = sorted(gold & removed)
        if orphaned:
            out[card] = orphaned
    return out


ADVICE = ("skill-path: a path named this way turns tests/test_skill_tool_names.py or "
          "tests/test_bench_audit_tasks.py red on main. If the file was removed or "
          "never tracked, name it in prose without a repo-rooted path (\"the code "
          "tree's stray `.t` scratch directory\"), then amend; if it should exist, "
          "it has to land first. Do not add it to PATH_KNOWN_UNFIXED or to a gold set.")

#: Deliberately not `skill-path FINDING:`. A finding says reword the SKILL.md and
#: brings ADVICE with it; this says edit a bench card too, and the two asks about one
#: commit must not arrive under one prefix — or be counted as one verdict.
GOLD_PREFIX = "skill-path GOLD REFRESH:"

GOLD_ADVICE = (
    "drop the item from that card's gold_items in this same commit — the gold set is "
    "hand-maintained, so the skill edit moved the corpus and nobody moved the card "
    "(unless another active skill still names the path, in which case the row is "
    "right and the card stays). "
    "tests/test_bench_audit_tasks.py::test_skill_dead_path_gold_still_matches_the_"
    "skills asserts the two sets are equal and goes red on the next live_vault run.")


def report(repo: Path) -> int:
    """Print the rung. Always returns 0: this reports, it does not block."""
    try:
        guard = _load_guard()
    except Exception as exc:                       # noqa: BLE001 - never block a commit
        print(f"skill-path: CHECK SKIPPED (cannot load {GUARD.name}: {exc})", flush=True)
        return 0
    try:
        findings, checked = path_findings(repo, guard)
        refresh = gold_refresh(repo, guard)
    except Exception as exc:                       # noqa: BLE001 - never block a commit
        print(f"skill-path: CHECK FAILED ({exc})", flush=True)
        return 0
    print(f"skill-path: {checked} staged skill/task file(s) checked, "
          f"{len(findings)} new dead path(s)", flush=True)
    for finding in findings:
        print(f"skill-path FINDING: {finding}", flush=True)
    if findings:
        print(ADVICE, flush=True)
    for card, rows in sorted(refresh.items()):
        listed = ", ".join(f"`{row}`" for row in rows)
        print(f"{GOLD_PREFIX} {card} still lists {listed} in gold_items, which this "
              f"commit takes out of the skills; {GOLD_ADVICE}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("VAULT_DIR", str(vault_root())),
                        help="git repo to inspect (default $VAULT_DIR or ~/obsidian)")
    args = parser.parse_args(argv)
    return report(Path(args.repo).expanduser())


if __name__ == "__main__":
    sys.exit(main())
