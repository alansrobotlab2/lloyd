#!/usr/bin/env python3
"""Refuse a commit whose OWN added lines put a wrong-zone clock into a skill template.

The rule is `scripts/skill_timezone.py`'s — one definition, three readers now. Two of
them predate this file and neither of them stands at this door:

  * ``scripts/automod/vault_round.py::skill_timezone_errors`` (the land rail), whose
    selector is ``p.startswith("skills/") and p.endswith("/SKILL.md")`` — a DETAIL file
    inside a skill directory never reaches it;
  * ``tests/test_skill_timezone_literals.py``, the reporting copy, whose live-vault
    ``_active_skills()`` reads ``*/SKILL.md`` for the same reason.

So the one live violation in the vault today is invisible to both. It is not in a
SKILL.md at all: it is ``skills/nightly-reflection-knowledge-write/step-1-and-commit-detail.md``
line 110, which interpolates ``$(date +%Y-%m-%d)`` into a ``vault-commit.sh`` subject —
the exact shape #2454 exists to refuse, written in the instructions of the job that
writes skills. Measured over ``~/obsidian/skills/**/*.md`` on 2026-11-07: 1 violation
outside ``skills/.archived/`` (that one line) and 14 violations across 8 archived
SKILL.md files, out of 567 ``.md`` files under ``skills/``. Re-measure with
``template_clock_violations`` over those paths.

Why a call site here, when the land rail already refuses this class: the nightly writer
goes out through ``scripts/util/vault-commit.sh``, which has its own ``git add`` and its
own ``git commit`` and is never validated by ``vault_round.validate()``. The class
therefore entered that door unrefused, and clause 5 of the filing keeps it that way for
now: widening the LAND rail to detail files would refuse every round touching the file
above, because that check is whole-body and line 110 is in its body. Diff-relativity is
the only shape that can exist on this door first, and this module is that shape.

Why diff-relative rather than a scan of every touched file: the tree already carries
violations this wrapper commits every cycle. A whole-body check on touched files would
freeze the nightly-reflection batch — which spans ``memory/ lloyd/ people/ projects/
knowledge/ skills/ backlog/ autonomy/`` — on somebody else's line, and that is #2245's
stranded-snapshot failure mode, not a rail. So the question this module asks of each
violation is the one #3545 and #3589 ask of their own classes: **did THIS commit write
that line?** A violation on a line the commit neither adds nor modifies is context, and
context commits.

How "this commit wrote that line" is measured, and why each step is git's answer and not
mine:

  * the file set and its status come from ``git diff --cached --name-status -z -M --
    skills/`` — NUL-separated so a path with a space or a quote cannot be read as two
    records, the way ``vault-commit.sh``'s own staging step learned (#1867, #2245);
  * the line numbers come from ``git diff --cached -U0 -M`` hunk headers, taking the
    NEW file's side: with zero context lines every line inside a ``+`` range is a line
    this commit wrote. Both pathspecs of a rename are passed to that call, because
    naming only the destination makes git answer "new file, every line added" and a
    ``git mv`` of a file that already carried a violation would then be refused for
    content nobody typed — measured, not assumed;
  * the CONTENT comes from ``git show :<path>``, the index copy, never the working
    tree: this wrapper commits pathspec-scoped batches (#1867, #2245), so a fix that is
    dirty-but-unstaged is not in the commit, and reading the tree would let the rung
    bless a commit that still ships the bad line.

The rule is then asked ONE LINE AT A TIME, and only about the lines in that set.
``template_clock_violations`` judges every line of a body independently, so a one-line
body answers exactly what a whole-body run would answer for that line — and it is the
line number's OWNER that stays on this side of the seam: the ``line N:`` the function
puts in its message is the number inside the body it was handed, which for a one-line
body is always 1 and would be a lie about the file. The message's reason is reused
verbatim, because the remedy it names is part of the rule and a copy here would drift.

Exit codes are the family's, and the polarity is the reason they matter: 0 when no line
this commit wrote carries the defect, exactly 1 when at least one does (a counting exit
would read as a broken check to the bash door), and 3 when the check could not look at
all. Anything the module cannot decide raises :class:`ClockCheckUnavailable` and becomes
a 3, never a 0 and never a 1 — a 1 from a crash would refuse every vault commit, and a 0
from a blind check would look exactly like a clean vault.

Usage::

    skill_timezone_findings.py [--vault DIR] [--json]

On a refusal, stdout carries the refused paths, one per line and nothing else, which is
what ``vault-commit.sh`` pulls out of the index; the refusal prose is all on stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Started BY PATH by the commit hook, which puts `scripts/util/` on `sys.path[0]` and
# not the checkout (#3403). Nothing at checkout-root scope is imported at module level:
# a top-level `ModuleNotFoundError` here would exit 1, and 1 is this tool's REFUSAL
# code, so a broken import would refuse every vault commit while naming no path. The
# rule is loaded lazily inside `main`'s guard instead, so the same failure is a 3.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: The governed tree, vault-relative, and the only extension in scope. Every `.md`
#: under it, so a SKILL.md and a step-detail file are judged the same way — the land
#: rail's `*/SKILL.md` selector is exactly the hole this rung closes (#3881 clause 3).
SKILLS_DIR = "skills"
MD_SUFFIX = ".md"

#: Where the one definition lives. Loaded by path, not by package name: this module is
#: started from `scripts/util/` by a bash door that has no interpreter to share, and a
#: namespace-package lookup would be a second way for the same file to fail to load.
_RULE_PATH = _REPO_ROOT / "scripts" / "skill_timezone.py"

#: `@@ -a,b +c,d @@`, the hunk header of `git diff -U0`. `b` and `d` are omitted when
#: the count is 1 (`@@ -2 +2 @@`), and `d` is `0` for a pure deletion, which is why the
#: capture groups are optional and the range is built from the NEW side only.
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)

#: The body-relative numbering `template_clock_violations` prefixes its reasons with. It
#: belongs to the one-line body this module hands it, never to the file, so it is
#: stripped and the caller's own line number is printed in its place.
def _reason_of(path: str, message: str) -> str:
    """The rule's reason, with its body-relative `line 1:` prefix taken off.

    Deliberately a prefix test and not a regex over the message: if the rule's wording
    ever changes, the worst this can do is print the file's line number twice. The
    alternative — parsing the NUMBER out of the prose — would silently drop every
    violation the day someone reflows a sentence, and a rung that quietly stops refusing
    is the failure this whole family is written against.
    """
    prefix = f"{path}: line 1: "
    return message[len(prefix):] if message.startswith(prefix) else message


class ClockCheckUnavailable(Exception):
    """The instrument could not run — exit 3, never a clean 0 and never a refusal."""


def _git(vault: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(vault), *args],
                          capture_output=True, text=True)


def _git_bytes(vault: Path, *args: str) -> subprocess.CompletedProcess:
    """A blob read in bytes. `text=True` would translate CRLF to LF and move every
    line number below the point of the first CRLF, so line N would name a different
    line than the one `git diff` reported."""
    return subprocess.run(["git", "-C", str(vault), *args], capture_output=True)


def _load_rule():
    """`scripts/skill_timezone.py`, the one definition of the class."""
    import importlib.util

    if not _RULE_PATH.exists():
        raise ClockCheckUnavailable(f"no rule module at {_RULE_PATH}")
    spec = importlib.util.spec_from_file_location("skill_timezone_for_clock_rung",
                                                 _RULE_PATH)
    if spec is None or spec.loader is None:
        raise ClockCheckUnavailable(f"cannot load {_RULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — reported, never laundered into a 0
        raise ClockCheckUnavailable(
            f"{_RULE_PATH.name}: {type(exc).__name__}: {exc}") from exc
    if not hasattr(mod, "template_clock_violations"):
        raise ClockCheckUnavailable(f"{_RULE_PATH.name} has no template_clock_violations")
    return mod


def _staged_blob(vault: Path, rel: str) -> str | None:
    """One path's INDEX content, or None when the index does not hold it.

    `:path` and not `rev=":"`, whose doubled `::path` `git show` rejects — the silent
    failure `ledger_strand_findings._blob` records: the read answers None, the caller
    reads "nothing there", and a check that never looked at a byte reports a clean tree.
    None means the path is not in the index at all, which for a commit here means a
    deletion, and a deletion adds no line.
    """
    r = _git_bytes(vault, "show", f":{rel}")
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", "replace")


def _staged_changes(vault: Path) -> list[tuple[str, str | None, str]]:
    """`(status_letter, old_path_or_None, new_path)` for what the INDEX differs on.

    `-M` keeps a rename one record instead of a delete plus an all-new add
    (`_added_new_lines` takes the same flag, and the two calls have to agree about what
    one change is); `-z` keeps a path with a space in it one path; the `skills/` pathspec
    is the whole scope clause, so a commit touching nothing under it costs one git call.

    An unborn HEAD needs no special case, and this is measured rather than assumed:
    `git diff --cached` against no HEAD compares the index with the empty tree, so an
    initial vault commit reports every staged file as `A` with every line inside a
    `+` range — which is the honest answer, because an initial commit really does write
    every line in it.
    """
    r = _git(vault, "diff", "--cached", "--name-status", "-z", "-M", "--",
             SKILLS_DIR + "/")
    if r.returncode != 0:
        raise ClockCheckUnavailable(
            f"git diff --cached --name-status: {(r.stderr or r.stdout).strip()[:200]}")
    toks = r.stdout.split("\0")
    out: list[tuple[str, str | None, str]] = []
    i = 0
    while i < len(toks):
        status = toks[i]
        i += 1
        if not status:
            break                       # the NUL that terminates the record stream
        letter = status[0]
        if letter in ("R", "C"):
            # Rename and copy records carry FROM then TO. Missing operands mean the
            # stream is not the shape documented, which is a broken read, not a 0.
            if i + 1 >= len(toks):
                raise ClockCheckUnavailable(f"truncated name-status record: {status}")
            out.append((letter, toks[i], toks[i + 1]))
            i += 2
        else:
            if i >= len(toks):
                raise ClockCheckUnavailable(f"truncated name-status record: {status}")
            out.append((letter, None, toks[i]))
            i += 1
    return out


def _in_scope(path: str) -> bool:
    return path.startswith(SKILLS_DIR + "/") and path.endswith(MD_SUFFIX)


def _added_new_lines(vault: Path, specs: list[str]) -> set[int]:
    """New-file line numbers this commit ADDs or MODIFIES, per git's own hunk headers.

    Both spellings of a rename are passed so git can pair them: naming only the
    destination answers `new file mode` with every line in a `+` range, which would
    refuse a `git mv` of a file that already carried the violation — the pre-image case
    clause 3 exists to keep committable.
    """
    r = _git(vault, "diff", "--cached", "-U0", "-M", "--", *specs)
    if r.returncode != 0:
        raise ClockCheckUnavailable(
            f"git diff --cached -U0: {(r.stderr or r.stdout).strip()[:200]}")
    out: set[int] = set()
    for m in _HUNK_RE.finditer(r.stdout):
        start = int(m.group(1))
        count = int(m.group(2) or 1)
        out.update(range(start, start + count))
    return out


def fresh_clock_violations(vault: str | Path) -> list[dict]:
    """Violations on lines this commit writes. `[]` means this commit is clean.

    `[{"path", "line", "status", "detail"}, ...]`, in path order then line order.
    `status` is git's porcelain letter for that path (`A`/`M`/`R`/`C`/`T`), which is what
    makes a refusal's "this commit wrote it" checkable from the record alone.

    Raises :class:`ClockCheckUnavailable` when the check itself cannot run or when its
    line model disagrees with git's; it never returns a partial answer, because a rung
    that silently skipped a file is the blindness this family calls a clean tree.
    """
    v = Path(vault)
    rule = _load_rule()
    out: list[dict] = []
    for letter, old, new in _staged_changes(v):
        if letter == "D" or not _in_scope(new):
            continue
        staged = _staged_blob(v, new)
        if staged is None:
            continue                    # not in the index: this commit deletes it
        added = _added_new_lines(v, [s for s in (old, new) if s])
        if not added:
            continue                    # a pure rename, or a mode-only change
        lines = staged.split("\n")
        for n in sorted(added):
            if n > len(lines):
                raise ClockCheckUnavailable(
                    f"{new}: git reported line {n} in a {len(lines)}-line index blob")
            for message in rule.template_clock_violations(new, lines[n - 1]):
                out.append({"path": new, "line": n, "status": letter,
                            "detail": _reason_of(new, message)})
    return out


def refusal_text(rec: dict) -> str:
    """The one refusal line for one violating line.

    It has to carry the path, the line, the rule's own reason (which is where the
    remedy for each of the two shapes lives) and the UTC spellings, or the caller cannot
    act on it from the message alone. Column 0 on purpose (#1070), and built in one
    place so a second door cannot describe the same refusal differently.
    """
    return (f"SKILL CLOCK LITERAL REFUSED: {rec['path']}: line {rec['line']} — this "
            f"commit {'adds' if rec['status'] in 'AC' else 'modifies'} a line whose "
            f"template states a clock in the wrong zone. {rec['detail']} "
            f"Remedy — spell the clock UTC in that line: a commit subject reads "
            f"`$(date -u +%F)`, not `$(date +%F)` (the subject's own date is UTC, so a "
            f"local `date` carries the previous calendar day either side of UTC "
            f"midnight); a displayed time derives its abbreviation from a clock "
            f"(`TZ=America/Los_Angeles date '+%H:%M %Z'`) or emits an ISO-8601 Z stamp "
            f"(`date -u +%Y-%m-%dT%H:%M:%SZ`). A violation this commit did NOT write is "
            f"not refused: only the lines in this commit's diff are judged. Nothing here "
            f"is deleted — the file's bytes stay, so edit that line and stage it again "
            f"(#3881).")


def findings(vault: str | Path) -> tuple[list[dict], list[str], str]:
    """`(records, refusal_lines, why_unread)` — the triple both doors act on.

    Shaped like `orphan_topic_findings.findings`. `why_unread` non-empty means the check
    did not run, and a door that commits anyway has to say so out loud.
    """
    try:
        records = fresh_clock_violations(vault)
        return records, [refusal_text(r) for r in records], ""
    except ClockCheckUnavailable as exc:
        return [], [], f"skill clock-literal check unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 — a rung that crashes is reported, not fatal
        return [], [], f"skill clock-literal check failed: {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    # `__doc__ or ""` because `python -OO` strips docstrings, and the AttributeError
    # that raises would reach the caller as a bare traceback whose exit code is 1 — the
    # code this tool returns for a REAL refusal.
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--vault", default=None,
                    help="vault root (default $LLOYD_VAULT, else the configured vault)")
    ap.add_argument("--json", action="store_true",
                    help="print the findings to stdout instead of the refusal block")
    args = ap.parse_args(argv)
    try:
        records = fresh_clock_violations(_resolve_vault(args.vault))
    except ClockCheckUnavailable as exc:
        print(f"SKILL CLOCK-LITERAL CHECK FAILED (not a clean read): {exc}",
              file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 — same reason: 1 would read as a refusal
        print(f"SKILL CLOCK-LITERAL CHECK FAILED (not a clean read): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    if not records:
        return 0
    for rec in records:
        print(refusal_text(rec), file=sys.stderr)
    print(f"SKILL CLOCK LITERAL: {len(records)} line(s) this commit writes state a "
          f"clock in the wrong zone, in {len({r['path'] for r in records})} file(s).",
          file=sys.stderr)
    if not args.json:
        # The contract with `vault-commit.sh`: stdout carries the refused paths and
        # nothing else in this mode, so the door can pull them out of the index. A
        # refused line left in the index becomes the NEXT commit's line, and the next
        # job's snapshot gets refused for prose it never wrote (#3545's reasoning).
        for path in sorted({rec["path"] for rec in records}):
            print(path)
    return 1


def _resolve_vault(arg: str | None) -> Path:
    """`--vault`, else `$LLOYD_VAULT`, else the configured vault root.

    No `~/obsidian` fallback: guessing a vault is how a check runs against the wrong
    tree and refuses a commit for lines that are not in it. Unresolvable is a 3.
    """
    if arg:
        return Path(arg)
    env = os.environ.get("LLOYD_VAULT")
    if env:
        return Path(env)
    from app.data_root import vault_root  # noqa: PLC0415 — see the module header
    return Path(vault_root())


if __name__ == "__main__":
    raise SystemExit(main())
