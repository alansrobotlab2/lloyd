#!/usr/bin/env python3
"""Name every bench card THIS commit ADDS whose stem has no assertion set behind it.

`app/harness/bench_authoring.py` refuses a keyless numbered bench card on four writer
lanes, and since #3870 (`b6feeef0`) `scripts/automod/vault_round.py::land()` refuses it at
the landing door too. Neither of them is this door. `scripts/util/vault-commit.sh` is the
OTHER way onto the vault's main — a `git add -A` of whatever the tree holds, run nightly
by `autonomy-data-pipeline` and the reflection jobs under the subject
`pre-flight … (unattributed dirty state)` — and it carried five keyless cards in
unrefused: `d1c89cf81`, `bf8be644e`, `dc72ec91a` (bench_024–028, all three commits dated
2026-10-04/05, all three confirmed present in the vault's history by #3974's triage). The
two doors disagreed on the same input, measured rather than inferred: the sweep door
committed `lloyd/bench/bench_99_premise_repro.md` with exit 0 while
`bench_card_landing_defect` handed that identical path a refusal. A guard on only one of
two write surfaces is not a guard (#2468), which is the class rule #3870's own land-rail
comment cited when it said it "does NOT close `scripts/util/vault-commit.sh`, the second
door". This closes it.

WHAT IT DOES IS THE LAND DOOR'S DECISION, NOT A SECOND ONE. Every verdict comes out of
`RAIL.bench_card_landing_defect(path, exists_at_head=…, table_path=…)` and every
bench-shaped stem out of `RAIL.BENCH_TASK_FILE_RE`, imported from
`app.harness.bench_authoring` — the same two calls
`scripts/automod/vault_round.py::_bench_card_assertion_errors` makes, in the same order,
with the same one-way `exists_at_head`. This module restates no stem pattern and reads no
assertion table: a stem rule or a table read cloned into the second door is a rule that
drifts from the one the writers and the land apply, and the day the two disagree the
"covered over both doors" claim is worth whatever the weaker of the two copies happens to
say. `lloyd/bench/` holds its sub-index `index.md` and its triage notes beside its cards,
so the stem test has to be the rail's.

SCOPE IS ONE COMMIT'S OWN STAGED ADDS: `git diff --cached --diff-filter=A` limited to
`lloyd/bench/`. Never a whole-tree count. The distinction is what keeps this off the
nightly-wide-red line #2415's owed ruling 2 withheld: this wrapper commits other writers'
dirty state, so a count over the tree would refuse a batch for a card its author never
touched — and a card that is already at HEAD is never newly refused even with its key
missing, because refusing an edit would make every existing card unfixable by precisely
the commit that would fix it. Both arms are the rail's own, reached through
`exists_at_head` from `git cat-file -e HEAD:<path>`.

DROPPED, NOT REFUSED — and that is the one deliberate difference from the orphan-topic and
timezone rungs beside this one, so it is stated in the diff rather than discovered later.
Their remedy is satisfiable from inside the vault (stage the `lloyd/MEMORY.md` hook line:
a vault file, in the same commit). This one's remedy is not: the assertion table is
`eval/autoresearch_assertions.yaml` in the LLOYD REPO, and a nightly vault snapshot can
neither carry it nor wait for it. `git reset -q -- <path>` then `exit 1` would leave the
card dirty, the next bare invocation would re-stage it, and every pre-flight commit from
then on would be refused for a file no job inside the vault can fix. So this rung takes
the shape the SQLite sidecar rung takes (#3236): pull the path out of the index, say so
loudly at column 0, and let the rest of the commit happen. The card can never reach main
through this door while it is keyless — which is the property #3974 asks for — and the
cost is that a permanently keyless card is dropped repeatedly rather than screaming, which
is what the post-landing owed check watches over real nightly traffic.

FAILS CLOSED ON ITS OWN MISSING RAIL, which is the other deliberate divergence, and the
four siblings fail open by ruling (`vault-commit.sh` prints `CHECK FAILED (exit N);
committing whatever the index holds` for them). An unimportable
`app.harness.bench_authoring` is not one of those: it is the guard itself gone blind, it
is never a writer's fault, and it is never transient, so committing through it is the
blind-check-reads-clean failure this family of rungs exists to refuse. Exit 4 says exactly
that and the door refuses the commit. Every OTHER unavailable state (exit 3 — the vault
would not answer `git diff --cached`) keeps the siblings' fail-open branch, because that
one can be a transient git condition and a freeze here is a nightly-wide red.

BECAUSE THAT FREEZE IS THE WHOLE POINT, "unimportable" is not allowed to mean "this door
picked an interpreter without the app's dependencies". `app/harness/__init__.py` eagerly
imports `app.harness.loop`, which imports `httpx`; the rail module this file needs is
stdlib-only (`logging/os/re/pathlib`). The door's own `$PY` resolution ends at
`command -v python3`, and a caller that sets no `LLOYD_PYTHON` — most of this file's
sibling tests, and any hand invocation — therefore gets an interpreter that cannot import
the package the rail lives in. Freezing every vault commit over a packaging accident would
be a worse failure than the one being closed, so a `ModuleNotFoundError` re-execs this
script ONCE under a venv candidate (the door's own list, plus the main checkout's venv
located through `git rev-parse --path-format=absolute --git-common-dir`, which is what
makes an automod worktree find the live `.venvs`), says one visible line about which
interpreter now answers, and only then, if that copy fails too, exits 4. A second
`ModuleNotFoundError` after the flag is set is the real missing rail; a `SyntaxError` or a
lost attribute never re-execs at all, because re-running the same defect under a better
interpreter is how a loop hides.

Usage::

    bench_assertion_findings.py [--vault DIR] [--json]

Exit 0 when nothing is dropped; 1 when at least one card is, with the dropped paths on
stdout (one per line, and nothing else in that mode) for the door to pull out of the index
and its reasons on stderr; 3 when the check could not read the vault; 4 when
`app.harness.bench_authoring` could not be imported. Both codes past 1 are greater than 1
by design: 1 is the refusal and anything above it is the instrument, which is the contract
every rung in this family keeps with its bash caller.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Run by path (`python …/scripts/util/bench_assertion_findings.py`), which puts this
# file's own directory on `sys.path[0]` and not the checkout — the insertion
# `scripts/util/orphan_topic_findings.py:98` and `ledger_strand_findings.py:111` each make
# for the same reason, and `tests/test_no_runtime_paths_in_code.py`'s rung node is the
# check that a rung starting by path can still reach `app.*`.
#
# The insertion has to precede every `app` read in this file, and NO `app` import may sit at
# module scope — not even the stdlib-only `app.data_root` the vault fallback needs. The
# reason is the door's exit-code contract, not tidiness: a caller that runs this file from a
# tree with no `app` package beside it must reach `_load_rail`, report exit 4 and stop the
# commit. A module-scope import dies at the import line instead, and the interpreter's own
# exit code there is 1 — the code that MEANS "drop these paths" — with empty stdout, so the
# door would take its drop branch, unstage nothing, print no drop line, and commit the card:
# a check that could not run laundering exactly what it exists to stop. Both imports are
# therefore inside the function that needs them, below this insertion.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Where the cards live IN THE VAULT. A git path, because every read below is a git read.
#: The `.md`-under-here filter is scope, not the stem rule: the land door filters the same
#: way (`p.startswith("lloyd/bench/") and p.endswith(".md")`) and THEN asks the rail.
BENCH_DIR = "lloyd/bench"

#: Exit codes. 1 is a drop, 3 is an unreadable vault, 4 is a missing rail module — see the
#: module docstring for why the last two are split.
EXIT_DROPPED = 1
EXIT_UNAVAILABLE = 3
EXIT_NO_RAIL = 4

#: The module whose verdict this is. Named as a string so the refusal text can name it
#: even when importing it is what failed.
RAIL_MODULE = "app.harness.bench_authoring"

#: Set once this file has re-exec'd itself, so a failing import can re-exec exactly once
#: and then report. Never an argument: an argument would make the loop-guard something a
#: caller can turn off, and an unbounded re-exec loop under a door that fails closed is a
#: door that never commits.
_REEXEC_FLAG = "LLOYD_BENCH_RAIL_REEXEC"

#: Candidate interpreters, in the order the DOOR itself tries them.
#: A re-exec hands over the SAME `__file__` (`os.execv` below), so the rail module the
#: child imports is still the one under the tree this script was invoked from — the
#: `_REPO_ROOT` on `sys.path` is this file's own ancestor, not the venv's. Only
#: site-packages come from elsewhere, which is the whole point: an automod gate runs this
#: file out of a worktree whose `.venvs` does not exist, and the answer must be the
#: worktree's `app/harness/bench_authoring.py` measured with dependencies that import.
#: (`vault-commit.sh`'s `$PY` resolution, #1237 then #2224), plus the one the door cannot
#: reach: the venv of the MAIN checkout, found through git rather than guessed from a
#: layout — `git rev-parse --path-format=absolute --git-common-dir` in an automod worktree
#: answers the live checkout's `.git`, whose `.venvs/lloyd` is the interpreter that has the
#: app's dependencies. Needed because `app/harness/__init__.py` eagerly imports
#: `app.harness.loop`, which imports `httpx`: the RAIL module itself is stdlib-only
#: (`logging/os/re/pathlib`), so the only thing standing between `/usr/bin/python3` — the
#: door's last-resort `$PY`, and the interpreter most of this file's sibling tests get —
#: and the rail is a package `__init__` that pulls in an HTTP client. `_venv_candidates`
#: is therefore not a convenience: without it, a rung that fails closed on a missing rail
#: would freeze every vault commit made by a caller that did not set `LLOYD_PYTHON`, which
#: is the nightly-wide red #2415's owed ruling 2 withheld and #3870's ruling named by hand.
def _venv_candidates() -> "list[Path]":
    out: "list[Path]" = []
    env = os.environ.get("LLOYD_PYTHON", "").strip()
    if env:
        out.append(Path(env))
    out.append(_REPO_ROOT / ".venvs" / "lloyd" / "bin" / "python")
    out.append(_REPO_ROOT.parent / "lloyd" / ".venvs" / "lloyd" / "bin" / "python")
    try:
        r = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "--path-format=absolute",
                            "--git-common-dir"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            common = Path(r.stdout.strip())
            if common.name == ".git":
                out.append(common.parent / ".venvs" / "lloyd" / "bin" / "python")
    except Exception:  # noqa: BLE001 — a layout probe is never a verdict
        pass
    out.append(Path.home() / "lloyd" / ".venvs" / "lloyd" / "bin" / "python")
    seen, uniq = set(), []
    me = Path(sys.executable).resolve() if sys.executable else None
    for c in out:
        try:
            key = c.resolve()
        except Exception:  # noqa: BLE001
            continue
        if key in seen or (me is not None and key == me):
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


class BenchRailUnavailable(Exception):
    """`app.harness.bench_authoring` could not be imported or has no rail in it.

    Distinct from `BenchCheckUnavailable` because the door's answer differs: a vault that
    will not answer a `git diff` is committed through with a warning, exactly as the four
    sibling rungs are, while a missing rail stops the commit (exit 4)."""


class BenchCheckUnavailable(Exception):
    """The vault would not answer — exit 3, never a laundered clean 0."""


def _reexec_for_the_rail() -> "Path | None":
    """The first candidate interpreter that exists and is not this one, or None.

    Reached ONLY from a `ModuleNotFoundError`, which is the one failure shape that says
    "wrong interpreter" rather than "broken rail": a missing `httpx` under the door's
    last-resort `python3` is that shape, and a `SyntaxError` or an `ImportError` for a
    symbol the rail module itself lost is not — re-exec'ing on those would run the same
    defect under a better interpreter and read as an infinite loop.

    Says one visible line on stderr before handing over, because which interpreter answers
    is which `app.harness.bench_authoring` answers (#999's reasoning about a second source
    of truth), and a swap nobody can see is exactly that. Prints and does not raise: the
    caller is about to `os.execv`, whose only failure mode is an exception, and the line has
    to be readable when that happens. Says it under the word `bench_assertion_findings`,
    never `BENCH ASSERTION`: a caller that greps the door's stderr for the rung's decision
    must not find a bookkeeping line wearing the decision's prefix.
    """
    if os.environ.get(_REEXEC_FLAG):
        return None            # already tried; the second failure is the real one
    for cand in _venv_candidates():
        if not cand.is_file():
            continue
        print(f"bench_assertion_findings: this interpreter cannot import {RAIL_MODULE};"
              f" re-running under {cand}", file=sys.stderr)
        os.environ[_REEXEC_FLAG] = "1"
        return cand
    return None


def _load_rail():
    """The rail module itself: `app.harness.bench_authoring`.

    Imported lazily and by name so a failure is attributable to the rail rather than to
    this file, and checked for the two attributes the verdict is made of: a module that
    imports but has lost `BENCH_TASK_FILE_RE` or `bench_card_landing_defect` would
    otherwise raise `AttributeError` per path, and an `AttributeError` caught by a broad
    handler at the call site reads as "no defect" — a bench rail that reports everything
    clean. Nothing here falls back to a local pattern for the same reason: a fallback is
    a second rule wearing the first one's name.
    """
    try:
        from app.harness import bench_authoring as RAIL  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        rerun = _reexec_for_the_rail()
        if rerun is not None:            # never returns; the exec'd copy answers instead
            os.execv(str(rerun), [str(rerun), str(Path(__file__).resolve()),
                                  *sys.argv[1:]])
        raise BenchRailUnavailable(
            f"{RAIL_MODULE} is unimportable ({type(exc).__name__}: {exc})") from exc
    except Exception as exc:  # noqa: BLE001 — the door decides, so report and stop
        raise BenchRailUnavailable(
            f"{RAIL_MODULE} is unimportable ({type(exc).__name__}: {exc})") from exc
    for name in ("BENCH_TASK_FILE_RE", "bench_card_landing_defect", "TABLE_REL"):
        if not hasattr(RAIL, name):
            raise BenchRailUnavailable(f"{RAIL_MODULE} has no {name}")
    return RAIL


def _git(vault: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(vault), *args],
                          capture_output=True, text=True)


def _added_bench_paths(vault: Path) -> list[str]:
    """Every `.md` the INDEX adds under `lloyd/bench/`, and nothing else.

    `--diff-filter=A` is the whole scope clause. A rename is NOT folded into its source
    (`--no-renames`, the reason `ledger_strand_findings._changed_paths` gives): a card
    moved INTO `lloyd/bench/` is a new path HEAD has never held, which is exactly the case
    the rail exists for, and rename detection would report it as an `R` and hide it.

    A vault with no commit yet has no HEAD to diff against, so the added set is the whole
    index — an initial commit really does add every file in it, and the empty answer a
    sibling module gives there would be a fail-open in the one commit most likely to carry
    a first card. `orphan_topic_findings.added_topic_paths` has that arm; this copies it.
    """
    v = Path(vault)
    if _git(v, "rev-parse", "--verify", "-q", "HEAD").returncode != 0:
        r = _git(v, "ls-files", "--cached", "--", BENCH_DIR + "/")
        if r.returncode != 0:
            raise BenchCheckUnavailable(
                f"git ls-files: {(r.stderr or r.stdout).strip()[:200]}")
        raw = r.stdout.splitlines()
    else:
        r = _git(v, "diff", "--cached", "--name-only", "--no-renames",
                 "--diff-filter=A", "--", BENCH_DIR + "/")
        if r.returncode != 0:
            raise BenchCheckUnavailable(
                f"git diff --cached: {(r.stderr or r.stdout).strip()[:200]}")
        raw = r.stdout.splitlines()
    kept = set()
    for p in raw:
        p = p.strip()
        if p.startswith(BENCH_DIR + "/") and p.endswith(".md"):
            kept.add(p)
    return sorted(kept)


def _exists_at_head(vault: Path, rel: str) -> bool:
    """Whether HEAD already holds this path — the rail's one-way arm, read from the vault.

    `git cat-file -e HEAD:<path>` is the same test
    `scripts/automod/vault_round.py::_bench_card_assertion_errors` runs (`:2536`), and for
    the same reason: HEAD is readable here too, so the ADD/EDIT decision is made the same
    way at both doors. An unborn HEAD, and any git error, answers False, which is the
    direction that asks the rail rather than waving the card through.
    """
    return _git(vault, "cat-file", "-e", f"HEAD:{rel}").returncode == 0


def dropped_cards(vault: str | Path, *,
                  table_path: "str | os.PathLike | None" = None) -> list[dict]:
    """The cards this commit adds that the rail says may not enter.

    `[{"path", "exists_at_head", "text"}, ...]` — `text` is the rail's own sentence,
    verbatim, because the two doors must not describe the same defect differently.

    `table_path` defaults to the copy that sits beside this script
    (`_REPO_ROOT / RAIL.TABLE_REL`), not to the rail's own default of
    `judge.ASSERTIONS_PATH`: this door is started by path from whatever cwd a nightly job
    happens to have, and the table that answers has to be the one in the checkout the door
    runs from. `None` answers through the rail's own default, which is what the land door
    does after `assertion_table_for_session` has picked a round's copy — there is no
    session and no round at this door, so there is nothing to pick.

    Raises `BenchRailUnavailable` / `BenchCheckUnavailable`; never returns a partial
    answer, and never returns a drop list it built itself.
    """
    RAIL = _load_rail()
    v = Path(vault)
    added = _added_bench_paths(v)
    if not added:
        return []
    table = Path(table_path) if table_path is not None else (_REPO_ROOT / RAIL.TABLE_REL)
    out: list[dict] = []
    for path in added:
        # The stem test is the rail's, applied to the NAME WITH `.md` exactly as
        # `bench_card_landing_defect` applies it internally — filtered here rather than
        # left to the rail so a non-card path under `lloyd/bench/` (its `index.md`, a
        # triage note) never reaches a refusal line that would name it.
        if not RAIL.BENCH_TASK_FILE_RE.match(path.rsplit("/", 1)[-1]):
            continue
        exists = _exists_at_head(v, path)
        text = RAIL.bench_card_landing_defect(path, exists_at_head=exists,
                                              table_path=table)
        if text:
            out.append({"path": path, "exists_at_head": exists, "text": text})
    return out


def findings(vault: str | Path) -> tuple[list[dict], list[str], str, int]:
    """`(records, refusal_lines, why_unread, exit_code)` — the quadruple the door acts on.

    Shaped like `orphan_topic_findings.findings` with the exit code carried out so an
    in-process reader (the second door's tests, a future audit) gets the same answer the
    CLI prints. `why_unread` non-empty means the check did not run; only the exit-4 half
    of that stops the commit, and the door says which half it saw.
    """
    try:
        records = dropped_cards(vault)
        return records, [r["text"] for r in records], "", (
            EXIT_DROPPED if records else 0)
    except BenchRailUnavailable as exc:
        return [], [], f"bench assertion check unavailable: {exc}", EXIT_NO_RAIL
    except BenchCheckUnavailable as exc:
        return [], [], f"bench assertion check unavailable: {exc}", EXIT_UNAVAILABLE
    except Exception as exc:  # noqa: BLE001 — a rung that crashes is reported, not fatal
        return [], [], (f"bench assertion check failed: "
                        f"{type(exc).__name__}: {exc}"), EXIT_UNAVAILABLE


def main(argv: list[str] | None = None) -> int:
    # `__doc__ or ""` because `python -OO` strips docstrings, and the AttributeError that
    # would raise here reaches the caller as a bare traceback whose exit code is 1 — the
    # code this tool returns for a real drop. `orphan_topic_findings.main:394` has the same
    # guard for the same reason.
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--vault", default=None,
                    help="vault root (default $VAULT_DIR — this door's own name — else "
                         "app.data_root.vault_root(): $LLOYD_VAULT_ROOT, $LLOYD_VAULT, "
                         "~/obsidian)")
    ap.add_argument("--json", action="store_true",
                    help="print the findings to stdout instead of the dropped-path list")
    args = ap.parse_args(argv)
    # The root comes from the resolver, never from a home-joined `obsidian` spelled in this
    # file: `tests/test_no_runtime_paths_in_code.py::
    # test_no_module_outside_the_resolver_computes_the_vault_root_from_home` refuses a module
    # that derives the vault from a home directory without honouring both override names, and
    # the failure it describes is this rung's worst one — pointed at the live vault by a
    # caller who believes it sandboxed itself, it reads a clean index and launders the card
    # into main. `vault_root()` honours `LLOYD_VAULT_ROOT` then `LLOYD_VAULT`; `$VAULT_DIR` is
    # the name THIS door's script reads, so it comes first among the env names, and `--vault`
    # outranks them all — which is also why the resolver is imported only when nothing named
    # a root: the door's own invocation never needs it, and an `app` import on that path would
    # put a bookkeeping failure ahead of the rail's verdict (see the `sys.path` note above).
    vault = args.vault or os.environ.get("VAULT_DIR")
    if not vault:
        try:
            from app.data_root import vault_root  # noqa: PLC0415
            vault = vault_root()
        except Exception as exc:  # noqa: BLE001 — a root nobody named and none can derive
            print(f"BENCH ASSERTION CHECK FAILED (not a clean read): no vault root and the "
                  f"resolver is unreachable ({type(exc).__name__}: {exc})", file=sys.stderr)
            return EXIT_UNAVAILABLE
    vault = Path(vault)
    try:
        records = dropped_cards(vault)
    except BenchRailUnavailable as exc:
        print(f"BENCH ASSERTION CHECK FAILED (no rail): {exc}", file=sys.stderr)
        return EXIT_NO_RAIL
    except BenchCheckUnavailable as exc:
        print(f"BENCH ASSERTION CHECK FAILED (not a clean read): {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except Exception as exc:  # noqa: BLE001
        print(f"BENCH ASSERTION CHECK FAILED (not a clean read): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    if not records:
        return 0
    for rec in records:
        # The rail's own sentence, unedited — the same words the land door says (#3870),
        # which is the point of calling the rail instead of restating it. Its remedy names
        # a round; the caller of THIS door has no round, and the door's own column-0 line
        # is what tells it where the rows actually have to go.
        print(rec["text"], file=sys.stderr)
    print(f"BENCH ASSERTION: {len(records)} bench card(s) added under {BENCH_DIR}/ "
          f"with no assertion set behind their stem; they are out of this commit.",
          file=sys.stderr)
    if not args.json:
        # The contract with `vault-commit.sh`: stdout carries the dropped paths and nothing
        # else in this mode, so the door can pull them out of the index. A dropped ADD left
        # staged is not a drop — `git commit` commits the index, so the next job's snapshot
        # would carry the card and be blamed for it (#3545's reasoning).
        for rec in records:
            print(rec["path"])
    return EXIT_DROPPED


if __name__ == "__main__":
    raise SystemExit(main())
