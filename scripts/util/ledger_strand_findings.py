#!/usr/bin/env python3
"""Refuse a commit that removes a loaded index line and leaves its ledger row live.

`lloyd/MEMORY.md` and `lloyd/USER.md` are the loaded memory index: the prompt gets
their bullets verbatim, and `lloyd/memory/memory-md-ledger.md` /
`user-md-ledger.md` carry one row per bullet — the row holds the entry's full body,
its `why`, its `retire_when` and the command that re-checks it. The bullet is the
hook; the row is the record. Dream #47 (vault commit `085f714b`, 2026-10-29 11:17)
deduped seven loaded lines into `SOUL.md`/`USER.md` and its `--stat` names
`lloyd/MEMORY.md` and `lloyd/USER.md` and NEITHER ledger, which stranded seven
rows. A stranded row is a record whose hook is gone: the index no longer points at
it, so nothing can reach it, and it can never be re-checked or legitimately
retired again. #2415 is the item that established the disposition rule — re-anchor
the row at its surviving line, or archive it under `lloyd/reviews/` with a reason
naming the line that is gone — and #2212 is why the join is the shared anchor and
not a per-writer copy.

Why a CALL SITE and not prose. `scripts/memory/memory_ledger.py status` has printed
`orphan rows: 7` for a day, and `scripts/memory/validate_memory_index.py:112-124`
says its own counts are "REPORT-ONLY by decision, not by omission (#2415 owed
ruling 2)". That decision was correct and stands: a clause that turns a nonzero
whole-tree orphan count red fails every nightly while pre-existing rows sit
undispositioned, which is exactly what owed ruling 2 withheld. Landed `d13514fa`
(2026-10-08) printed the number three weeks before `085f714b` stranded seven rows,
so printing is proved insufficient. What was missing is a step that refuses the
DIFF — the guard/call-site end of the landed guard > template > prose
discriminator.

The check is therefore DIFF-RELATIVE and reads the INDEX, never the working tree and
never the whole-tree orphan count:

  * every path this commit actually changes (`git diff --cached --name-only`), so a
    commit that touches no loaded index line is never refused, whatever orphans the
    tree already carries — today's seven included;
  * for a changed loaded file, the entries at HEAD minus the entries in the index,
    which is the set of loaded lines THIS commit removes (a line whose text was
    reworded inside its 60-character anchor is in that set, because from the
    ledger's seat that row is exactly as stranded as a deleted one);
  * for each removed line, whether its row is still in the LEDGER'S INDEX COPY — so
    a commit that dispositions the row in the same staged diff is clean. That is the
    whole clause: the row must go, or move to `lloyd/reviews/<ledger>-archive.md`,
    in this commit, not "later".

One matcher, two doors (#2468 is what guarding one and calling it coverage costs):
`scripts/util/vault-commit.sh` runs this as a subprocess after its `git add` — that
door is bash and has no interpreter to share — and `scripts/automod/vault_round.py`'s
`land()` imports it, because that door has its own `git add` and its own `git
commit` and never calls the wrapper. `085f714b`'s own subject line ("merge the
vault-land pair") points at the second door. The shape is #3236's
`sqlite_sidecar_findings.py` pair, with one difference that is the whole item: a
sidecar is a stray file, so dropping it from the index is the right repair and the
commit proceeds; a stranded row is the commit's own defect, unstaging anything would
be silently rewriting another caller's commit, and only a refusal teaches it. So this
rung hard-refuses and touches nothing.

The one cost, stated so a curator is not surprised by it: an edit that removes a
loaded line and dispositions the row in a LATER commit is refused at whichever door
commits first, including by the nightly sweep, until both files are staged together.
That is the demand, not a bug in it — "later" is precisely what left seven rows
behind.

Nothing here is a policy the ledger does not already have: the entry set, the anchor
and the row set are `scripts/memory/memory_ledger.py`'s own `loaded_entries`,
`anchor_of` and `ledger_rows`, loaded by path, so the guard and `status()` cannot
disagree about what a row or an orphan is. This module does not call `status()`.

Usage::

    ledger_strand_findings.py [--vault DIR] [--json]

Exit 0 when no staged line is stranded; exactly 1 when at least one is, whatever the
count — `#3236`'s sibling returns `1 + n`, which its bash caller reads as "the rung
itself failed" (`vault-commit.sh` treats anything past 1 as a broken check), so a
counting exit code here would turn a genuine two-row refusal into a shrug; and 3 when
the check could not run at all (no git, no vault, no `memory_ledger.py` beside the
instrument), because the failure mode of every rung in this family is a check that
goes blind and looks like a clean tree, and a broken instrument must not be laundered
into a green one. Both doors hard-refuse the commit on 1 and proceed on 3 while
saying so. `main()` catches every exception, so an exit of 1 can only ever mean a
found strand.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
# Run by path (`python …/scripts/util/ledger_strand_findings.py`), which puts this file's own directory on
# `sys.path[0]` and not the checkout: without the insertion below, `app.data_root`
# raises `ModuleNotFoundError` and the reader silently never runs. Same reason as
# `scripts/skill_lint.py`'s insertion. (#3403)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from app.data_root import vault_root  # noqa: E402

# scripts/util/ -> scripts/ -> the lloyd checkout that carries the instrument.
_ROOT = Path(__file__).resolve().parents[2]
_LEDGER_SCRIPT = _ROOT / "scripts" / "memory" / "memory_ledger.py"

#: Where the loaded index files sit in the vault, and where their ledgers sit.
#: `memory_ledger.archive_path` hangs the archive off the same directory, which is
#: how the disposition named below stays the real one and not a restatement.
INDEX_DIR = "lloyd"
LEDGER_DIR = "lloyd/memory"


class StrandCheckUnavailable(Exception):
    """The instrument could not be read at all — exit 3, never a clean 0."""


def _load_memory_ledger():
    """`scripts/memory/memory_ledger.py` as a module, loaded by path.

    Function-local, the way `validate_memory_index.py:101-107` loads it for the same
    reason: this file is imported by `vault_round.py::land()`, which every vault
    landing runs, and an import-time failure there would be a land that cannot land
    rather than a rung that declined. The repo root goes on `sys.path` because the
    instrument imports `app.*`, and the bash door inherits whatever directory its
    caller happened to run from.
    """
    import importlib.util

    if not _LEDGER_SCRIPT.exists():
        raise StrandCheckUnavailable(f"no ledger instrument at {_LEDGER_SCRIPT}")
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location("memory_ledger_for_strand_rung",
                                                 _LEDGER_SCRIPT)
    if spec is None or spec.loader is None:
        raise StrandCheckUnavailable(f"cannot load {_LEDGER_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — reported, never laundered into a 0
        raise StrandCheckUnavailable(f"{_LEDGER_SCRIPT.name}: {type(exc).__name__}: {exc}") from exc
    return mod


def _git(vault: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(vault), *args],
                          capture_output=True, text=True)


def _blob(vault: Path, rel: str, *, staged: bool) -> str | None:
    """One path's content in the index (`staged=True`) or at HEAD, or None if absent.

    A keyword and not a `rev` string because the two spellings differ by one colon and
    a `rev=":"` argument would build the doubled `::path` that `git show` rejects —
    which is not merely an error, it is the silent kind: `_blob` answers None, the
    entry set comes back empty, no line counts as removed, and a guard that never
    looked at a byte reports a clean tree.

    None means the file is not in that tree — for HEAD, a loaded index or ledger this
    round adds; for the index, a file this commit deletes. Both are read as the empty
    text by the caller, which is the honest reading: no entries, and no rows.
    """
    r = _git(vault, "show", f"HEAD:{rel}" if not staged else f":{rel}")
    if r.returncode != 0:
        return None
    return r.stdout


def _changed_paths(vault: Path) -> set[str]:
    """Every path the INDEX differs from HEAD on — what the commit will contain.

    `--no-renames` is deliberate: rename detection reports a rename as
    delete-plus-add with no content, and the set difference of entry lists is what
    this check needs, so the pair has to stay two entries. Reading the index and not
    the working tree is what keeps a scoped commit (`-- <path>`) that leaves a
    loaded-file edit unstaged out of the refusal: that commit does not remove the
    line, so it owes nothing.
    """
    r = _git(vault, "rev-parse", "--verify", "-q", "HEAD")
    if r.returncode != 0:
        # A vault with no commit has nothing staged against it, so there is nothing
        # this commit could strand. Same answer as an empty diff, for a different
        # reason; `_blob(..., "HEAD", ...)` would fail loudly otherwise.
        return set()
    r = _git(vault, "diff", "--cached", "--name-only", "--no-renames")
    if r.returncode != 0:
        raise StrandCheckUnavailable(f"git diff --cached: {(r.stderr or r.stdout).strip()[:200]}")
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


def ledger_rel(index_name: str, ml) -> str:
    """`MEMORY.md` -> `lloyd/memory/memory-md-ledger.md`, off `ml.LEDGERS`.

    One map, read from the instrument that owns it (#2212), so a fifth loaded file or
    a renamed ledger moves this guard by itself rather than leaving it policing a
    file that no longer exists.
    """
    return f"{LEDGER_DIR}/{ml.LEDGERS[index_name]}"


def archive_rel(index_name: str, ml) -> str:
    """The archive a row may be MOVED to, as a vault-relative path.

    `ml.archive_path` is the one definition of where `retire` puts retired rows:
    `<memories_dir>/reviews/<ledger-stem>-archive.md`, and `memories_dir` is
    `INDEX_DIR` here, so passing it through yields the vault-relative spelling with no
    layout restated. Taking only the file's name and re-prefixing the index directory
    would drop the `reviews/` segment — `lloyd/reviews/` is where `retire` actually
    writes, and a route the message names but `retire` does not write to is a
    disposition the caller makes into a new orphan.
    """
    return ml.archive_path(Path(INDEX_DIR), ml.LEDGERS[index_name]).as_posix()


def stranded_rows(vault: str | Path) -> list[dict]:
    """Staged removals whose ledger row is still live, in index order.

    `[{"loaded", "ledger", "archive", "anchor", "line", "head", "staged"}, ...]` —
    `head`/`staged` are the loaded file's entry counts on each side of the diff, the
    numbers that make "this commit removed it" checkable rather than asserted.

    Raises `StrandCheckUnavailable` when the check itself cannot run. It never
    returns a partial answer: a guard that silently skipped a file would be the same
    blindness `status()`'s printed count already was.
    """
    v = Path(vault)
    ml = _load_memory_ledger()
    changed = _changed_paths(v)
    out: list[dict] = []
    for name in sorted(ml.LEDGERS):
        loaded_rel = f"{INDEX_DIR}/{name}"
        led_rel = ledger_rel(name, ml)
        if loaded_rel not in changed:
            # Diff-relative by construction: this loaded file is not in this commit,
            # so whatever orphans its ledger already carries are not this commit's.
            continue
        head_text = _blob(v, loaded_rel, staged=False)
        staged_text = _blob(v, loaded_rel, staged=True)
        # One measure on both sides: distinct non-empty anchors. `status()` counts rows
        # against anchors, not lines against lines, so reporting a line count on one
        # side and an anchor count on the other would print a `3 -> 2` that was partly
        # an artefact of a duplicated anchor rather than of a removal.
        head_anchors = {a for a in (ml.anchor_of(e)
                                    for e in ml.loaded_entries(head_text or "")) if a}
        staged_anchors = {a for a in (ml.anchor_of(e)
                                      for e in ml.loaded_entries(staged_text or "")) if a}
        line_of: dict[str, str] = {}
        for entry in ml.loaded_entries(head_text or ""):
            # First writer wins, the way `status()`'s `duplicate_anchors` treats a
            # repeated anchor: one row covers it, so one refusal names one line.
            line_of.setdefault(ml.anchor_of(entry), entry)
        removed = [a for a in sorted(head_anchors - staged_anchors)]
        if not removed:
            continue
        led_anchors = {r["anchor"] for r in
                       ml.ledger_rows(_blob(v, led_rel, staged=True) or "")}
        for anchor in dict.fromkeys(removed):
            if anchor not in led_anchors:
                # No row covers the removed line, so this commit strands nothing: it
                # is either a line that never had a row (an `unrepresentable` anchor,
                # or an entry the ledger predates) or a line already orphaned by an
                # earlier commit — which is #2415's backlog, not this diff.
                continue
            out.append({"loaded": loaded_rel, "ledger": led_rel,
                        "archive": archive_rel(name, ml), "anchor": anchor,
                        "line": line_of[anchor],
                        "head": len(head_anchors), "staged": len(staged_anchors)})
    return out


def refusal_text(rec: dict) -> str:
    """The one refusal line for one stranded row.

    Column 0 on purpose (#1070): the bash door's own no-change block has to carry its
    leading token there or `git commit`'s comment filter strips the sentence that
    names the reason. Both doors print text built here, so the two doors cannot
    describe the same refusal differently.
    """
    line = rec["line"].strip()
    if len(line) > 240:
        line = line[:240] + " […]"
    return (f"LEDGER STRAND REFUSED: {rec['loaded']} — this commit removes a loaded "
            f"index line and its row stays live in {rec['ledger']}. "
            f"Removed line: {line!r}. Anchor: {rec['anchor']!r}. "
            f"Disposition that row in this same commit — delete it, or move it to "
            f"{rec['archive']} under a reason naming the removed line (#2415) — and "
            f"commit the index and the ledger together (#3589). "
            f"Nothing is unstaged and nothing is committed.")


def findings(vault: str | Path) -> tuple[list[str], str]:
    """`(refusal_lines, why_unread)` — the pair both doors act on.

    `why_unread` is non-empty exactly when the check could not run, so a caller that
    commits anyway has to say out loud that it did so on a blind rung. An empty
    `why_unread` with no refusals is a real clean read, and that distinction is the
    only thing that keeps a skipped check from reading as a passing one.
    """
    try:
        return [refusal_text(r) for r in stranded_rows(vault)], ""
    except StrandCheckUnavailable as exc:
        return [], f"ledger strand check unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 — a rung that crashes is reported, not fatal
        return [], f"ledger strand check failed: {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    # `__doc__ or ""` because `python -OO` strips docstrings, and an AttributeError
    # raised here — before the vault argument is even parsed — would reach the caller as
    # a bare traceback whose exit code is 1, the same code this tool returns for a real
    # refusal. A stripped build would then refuse every vault commit that carries a
    # loaded index line, with a message quoting no line at all.
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--vault", default=None,
                    help="vault root (default $LLOYD_VAULT, else ~/obsidian)")
    ap.add_argument("--json", action="store_true",
                    help="print the findings to stdout instead of the refusal block")
    args = ap.parse_args(argv)
    vault = Path(args.vault or (__import__("os").environ.get("LLOYD_VAULT") or
                                (vault_root())))
    try:
        records = stranded_rows(vault)
    except StrandCheckUnavailable as exc:
        print(f"LEDGER STRAND CHECK FAILED (not a clean read): {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"LEDGER STRAND CHECK FAILED (not a clean read): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    if not records:
        return 0
    for rec in records:
        print(refusal_text(rec), file=sys.stderr)
    print(f"LEDGER STRAND: {len(records)} loaded index line(s) removed with their "
          f"ledger row still live.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
