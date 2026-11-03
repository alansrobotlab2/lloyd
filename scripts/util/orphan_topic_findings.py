#!/usr/bin/env python3
"""Refuse a commit that ADDS a memory topic file no index line links and no row disposes of.

`lloyd/MEMORY.md` is the loaded index and `lloyd/memory/<slug>.md` is the detail it
hooks with a `→ topics/<slug>` line. That hook is the ONLY route a topic file has into
a prompt — it is how `memory_read(file="topics/<slug>")` is reached — so a file on
disk no index line names is a rule that exists and is never applied. #2399 proved that
silently: dream #47 found six such files on 2026-10-07, five class rules plus
`memory-md-ledger`, unread by every prompt for a week while
`scripts/memory/validate_memory_index.py` printed `OK` nightly.

Why a CALL SITE at the commit boundary, when the rule already exists in prose and in a
validator. Three facts, each measured:

  * the prose rule (`skills/nightly-reflection-knowledge-write/SKILL.md` Step 2.5,
    "the index line goes in the same commit") did not hold: of the 29 topic-file adds
    under `lloyd/memory/` since 2026-09-25, 12 landed with no `→ topics/` line in
    THEIR OWN commit — 41%, 6 of them from `nightly-reflection: pre-flight …` sweep
    commits. Two vaults (`871da02a`, `3b962ad7`) committed while the validator was
    red;
  * the validator cannot do this job. Its unlinked arm is whole-tree and
    report-only-by-decision at the levels #2415 owed ruling 2 withheld, and its
    `ok:false` stopped nothing: the nightly committed anyway, twice;
  * 17 of those 29 adds already link in the same commit. The healthy writer exists
    and already does the work, which is what makes a commit-boundary refusal
    satisfiable rather than a nightly-wide red.

So the rule moves to the commit boundary and is scoped to what a commit can be held
for: **paths this commit ADDS** (`git diff --cached --diff-filter=A`). A commit that
carries a pre-existing orphan it did not create still commits, which is the difference
between this rail and the whole-tree count that owed ruling 2 correctly refused to
turn red. It is also what keeps the nightly sweep, which commits other writers' dirty
state, from freezing on somebody else's file — #2245's stranded-commit failure mode.

The index is read as `git show :lloyd/MEMORY.md`, the LEDGER'S OWN STAGED COPY of
`lloyd/MEMORY.md`, never the file on disk. This wrapper supports pathspec-scoped
commits (#1867 `f92b87d9`, #2245 `238c3bc8`) and `land()` stages only the paths it was
named, so a hook line that is dirty-but-unstaged is not in the commit and does not
cover anything: reading the working tree would pass a commit that leaves the orphan
red, which is the exact hole this item was filed to close.

The ledger arm is the second remedy, and it is deliberately narrow. Keying it on a
bare mention of the slug shelters an orphan whose disposition is itself in dispute:
`lloyd/reviews/memory-md-retired.md` names `replay-an-age-bearing-alarm-at-its-own-instant`
seven times, and five of those mentions are check commands and remediation prose that
spell the on-disk path (`ls ~/obsidian/lloyd/memory/replay-….md`,
`test -e …/lloyd/memory/replay-….md`) while the row's own §5 file-survival claim is
RECORDED FALSE at `:75`. So a substring match on the path would have sheltered the one
file whose survival the ledger says nobody verified. What counts instead is a FIELD: a
line of the form

    - **Topic file disposition:** `lloyd/memory/<slug>.md` — <what happened to it>

inside `lloyd/reviews/memory-md-retired.md`. Nothing in the live ledger carries that
field today, so the arm is empty and every unlinked add is refused — the safe
direction, and the only one this round can take: #3539, which owned the disposition
row format, closed `unnecessary` without ever building its validator-side helper (its
owed ruling then said explicitly that NO validator-side exemption is added), so there
is no helper to import and the field is owed-check's to rule on (#3545 owed ruling).
A refusal message therefore names the field it reads, so a writer can satisfy it.

ONE MATCHER, TWO DOORS, the #3236/#3589 shape. `scripts/util/vault-commit.sh` runs
this as a subprocess (that door is bash and has no interpreter to share);
`scripts/automod/vault_round.py::land()` imports it, because that door has its own
`git add` and its own `git commit` and never calls the wrapper. A guard on only one of
two write surfaces is not a guard (#2468), and the second door is the one the
`automod_vault_land` route goes through — `agent_mcp/vault.py` has no commit site at
all, which is why the item named it wrongly at filing.

Nothing here is a second policy: the link grammar is
`scripts/memory/validate_memory_index.py::links`, loaded lazily and by path exactly as
that file loads `memory_ledger`, so the rail and the nightly validator cannot disagree
about what a `→ topics/` link is.

Usage::

    orphan_topic_findings.py [--vault DIR] [--json]

Exit 0 when no added topic file is orphaned; exactly 1 when at least one is (a counting
exit code would read as "the rung itself broke" to the bash caller, which treats
anything past 1 as a failed check — #3236's sibling has that shape and this one does
not); 3 when the check could not run at all. Both doors refuse on 1 and proceed on 3
while saying so out loud, because the failure mode of every rung in this family is a
check that goes blind and looks like a clean tree.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Run by path (`python …/scripts/util/orphan_topic_findings.py`), which puts this
# file's own directory on `sys.path[0]` and not the checkout — same fix as #3403's
# insertion in `ledger_strand_findings.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from app.data_root import vault_root  # noqa: E402

#: Where the topic files live in the vault, where the index is, and where the
#: retirement ledger is. All three are vault-relative git paths, because every read
#: below is a git read.
TOPIC_DIR = "lloyd/memory"
MEMORY_REL = "lloyd/MEMORY.md"
LEDGER_REL = "lloyd/reviews/memory-md-retired.md"

#: The one exclusion in the added-path set. A `*-md-ledger.md` file is a LEDGER, not a
#: topic file: `scripts/memory/memory_ledger.py`'s `LEDGERS` map reads it by name for
#: `MEMORY.md` rows and nothing links it from an index line, so treating it as a topic
#: file would refuse every ledger write — including the disposition writes this rail's
#: own remedy asks for.
LEDGER_EXCLUDE_SUFFIX = "-md-ledger.md"

#: The ledger field that disposes of a topic file. See the module docstring: a bare
#: mention of the slug or the path is NOT coverage, because the live ledger mentions
#: the disputed `replay-an-age-bearing-…` path five times in check commands and
#: remediation prose whose survival claim is recorded false.
DISPOSITION_FIELD = "topic file disposition"

#: The same field as a human reads it. The comparison key above is lower-cased because
#: the label is compared case-folded, and the live ledger's own fields are title case
#: (`- **Ledger row:**`, `- **Check command:**`); a refusal that told a writer to write
#: `**topic file disposition:**` would be asking for a spelling no row in the file uses.
DISPOSITION_FIELD_LABEL = "Topic file disposition"

#: `- **Label:** rest` — the field shape every row in this ledger family uses, from
#: `**Ledger row:**` to `**Check command:**`. Only a line in that shape is read, and
#: only when its label IS the disposition field.
_FIELD_LINE_RE = re.compile(r"^\s*[-*]\s*\*\*\s*(?P<label>[^*]*?)\s*\*\*\s*:?\s*(?P<text>.*)$")
#: The on-disk path spelled out, inside a disposition line. `[^\s`…,;)]+` so a trailing
#: sentence or a backtick cannot be swallowed into the slug.
_PATH_RE = re.compile(r"(?:^|[\s`('\"=])" + TOPIC_DIR + r"/([^\s`'\"(),;]+)\.md(?![A-Za-z0-9./-])")

_VALIDATOR = _REPO_ROOT / "scripts" / "memory" / "validate_memory_index.py"


class OrphanTopicCheckUnavailable(Exception):
    """The instrument could not be read at all — exit 3, never a clean 0."""


def _load_links():
    """`validate_memory_index.links` — the one definition of a `→ topics/` link.

    Lazy and by path, the way that file loads `memory_ledger` for the same reason: this
    module is imported by `vault_round.py::land()`, which every vault landing runs, and
    an import-time failure there would be a land that cannot land rather than a rung
    that declined. The link grammar is imported rather than copied because the day the
    index's arrow spelling changes, a copy here would start refusing healthy commits —
    which is the one failure mode this rail can produce that no test would catch at
    the validator.
    """
    import importlib.util

    if not _VALIDATOR.exists():
        raise OrphanTopicCheckUnavailable(f"no index validator at {_VALIDATOR}")
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("validate_memory_index_for_orphan_rung",
                                                 _VALIDATOR)
    if spec is None or spec.loader is None:
        raise OrphanTopicCheckUnavailable(f"cannot load {_VALIDATOR}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — reported, never laundered into a 0
        raise OrphanTopicCheckUnavailable(
            f"{_VALIDATOR.name}: {type(exc).__name__}: {exc}") from exc
    return mod.links


def _git(vault: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(vault), *args],
                          capture_output=True, text=True)


def _staged_blob(vault: Path, rel: str) -> str | None:
    """One path's INDEX content, or None when the index does not hold it.

    `:path` and not `rev=":"`, whose doubled `::path` `git show` rejects — the silent
    failure recorded in `ledger_strand_findings._blob`: the read answers None, the
    caller reads "nothing links it", and a check that never looked at a byte reports a
    refusal. Here the polarity is inverted, so the same care matters the other way: an
    index line that exists but is read as absent would refuse a healthy commit.
    """
    r = _git(vault, "show", f":{rel}")
    if r.returncode != 0:
        return None
    return r.stdout


def _worktree_text(vault: Path, rel: str) -> str | None:
    """The ledger's content ON DISK, or None. Only ever the ledger — see
    `_dispositioned_slugs` for why the index is never read this way."""
    p = vault / rel
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _topic_adds(raw_paths: list[str]) -> list[str]:
    """Keep the `.md` files this commit adds under `TOPIC_DIR`, minus the ledgers.

    Every surviving path is kept, including one whose stem is not a legal slug: a
    nested or odd-named file is a file no index line can hook either, and dropping it
    here would be the silent skip this family of checks refuses to make.
    """
    out = []
    for p in raw_paths:
        p = p.strip()
        if not p.startswith(TOPIC_DIR + "/") or not p.endswith(".md"):
            continue
        if p.endswith(LEDGER_EXCLUDE_SUFFIX):
            continue
        out.append(p)
    return sorted(set(out))


def added_topic_paths(vault: str | Path) -> list[str]:
    """Every `lloyd/memory/*.md` the INDEX adds, and nothing else.

    `--diff-filter=A` is the whole scope clause: an untracked-then-committed file is an
    `A`, a modification of a file HEAD already holds is an `M`, and only the first is
    this commit's doing. `--no-renames` for the reason
    `ledger_strand_findings._changed_paths` gives — rename detection reports a move as
    a delete plus an add with no content, and a move out of the directory is not a new
    orphan.

    A vault with no commit yet has nothing staged AGAINST a HEAD to diff, so the added
    set is the whole index: an initial commit really does add every file in it. The
    empty-set answer the sibling module gives there would be a fail-open here, because
    the thing an initial vault commit most often adds is exactly a topic file.
    """
    v = Path(vault)
    if _git(v, "rev-parse", "--verify", "-q", "HEAD").returncode != 0:
        r = _git(v, "ls-files", "--cached", "--", TOPIC_DIR + "/")
        if r.returncode != 0:
            raise OrphanTopicCheckUnavailable(
                f"git ls-files: {(r.stderr or r.stdout).strip()[:200]}")
        return _topic_adds(r.stdout.splitlines())
    r = _git(v, "diff", "--cached", "--name-only", "--no-renames",
             "--diff-filter=A", "--", TOPIC_DIR + "/")
    if r.returncode != 0:
        raise OrphanTopicCheckUnavailable(
            f"git diff --cached: {(r.stderr or r.stdout).strip()[:200]}")
    return _topic_adds(r.stdout.splitlines())


def slug_of(path: str) -> str:
    """`lloyd/memory/foo.md` -> `foo`, the string an index line links as `→ topics/foo`.

    The path minus the directory and the extension, which is what `memory_read` takes
    and what the validator's `LINK_RE` captures. `lloyd/memory/` is flat on the live
    vault, so this is the whole mapping for every real file; a nested path would yield
    `sub/foo` and be looked for under `→ topics/sub/foo`, which is how
    `app/memory_ceiling.topic_path` resolves it too.
    """
    return path[len(TOPIC_DIR) + 1:-len(".md")]


def dispositioned_slugs(text: str) -> set[str]:
    """Every slug a `**Topic file disposition:**` row in the ledger disposes of.

    Field-keyed on purpose, and the check for it is in the module docstring: five of
    the seven mentions of the disputed `replay-an-age-bearing-…` path in the live
    ledger are check commands and remediation prose, so a substring match would
    shelter a file whose survival the ledger itself records as FALSE. A line that is
    not in the `- **<label>** …` shape, or whose label is not this field, names
    nothing.
    """
    out: set[str] = set()
    for line in text.splitlines():
        m = _FIELD_LINE_RE.match(line)
        if not m:
            continue
        # `rstrip(":")` because every row in this ledger family writes the colon INSIDE
        # the bold (`- **Ledger row:** …`, `- **Check command:** …`), so the captured
        # label carries it. Dropped rather than added to the constant so the field name
        # in the refusal text stays the spelling a writer copies.
        label = m.group("label").strip().rstrip(":").strip().lower()
        if label != DISPOSITION_FIELD:
            continue
        for pm in _PATH_RE.finditer(m.group("text")):
            out.add(pm.group(1))
    return out


def _ledger_text(vault: Path) -> str:
    """The ledger as this commit will leave it: staged copy, else the on-disk file, else empty.

    Read on disk when the index does not hold it because a disposition row is a row
    that may well be sitting unstaged in a tree the nightly already wrote — and unlike
    the index, an unstaged row strands nothing: the file it names is IN this commit
    either way, the ledger is not part of the commit's meaning, and #3535's ledger read
    is about the tree. Absent or unreadable is the empty text, which is the honest
    reading (no rows) and the direction that refuses rather than blesses: clause 4 of
    #3545 is that an absent ledger must still refuse an unlinked add, never crash and
    never silently skip.
    """
    staged = _staged_blob(vault, LEDGER_REL)
    if staged is not None:
        return staged
    return _worktree_text(vault, LEDGER_REL) or ""


def orphan_adds(vault: str | Path) -> list[dict]:
    """The added topic files this commit leaves unlinked and undispositioned.

    `[{"path", "slug", "memory_md", "ledger", "field", "linked"}, ...]` — `linked` is
    how many `→ topics/` slugs the STAGED index carries, the number that makes "the
    index has no line for it" checkable rather than asserted.

    Raises `OrphanTopicCheckUnavailable` when the check itself cannot run; it never
    returns a partial answer.
    """
    v = Path(vault)
    added = added_topic_paths(v)
    if not added:
        return []
    links = _load_links()
    memory_text = _staged_blob(v, MEMORY_REL)
    if memory_text is None:
        # No index in the commit at all: nothing links anything, so every add in scope
        # is orphaned. Empty text is the honest read of "no lines", and the refusal
        # then names the index as the file to stage, which is the right remedy.
        memory_text = ""
    linked = set(links(memory_text))
    disposed = dispositioned_slugs(_ledger_text(v))
    out: list[dict] = []
    for path in added:
        slug = slug_of(path)
        if slug in linked or slug in disposed:
            continue
        out.append({"path": path, "slug": slug, "memory_md": MEMORY_REL,
                    "ledger": LEDGER_REL, "field": DISPOSITION_FIELD,
                    "linked": len(linked)})
    return out


def refusal_text(rec: dict) -> str:
    """The one refusal line for one orphaned add.

    It has to carry three things or the caller cannot act on it: the offending path,
    both remedies, and the fact that the index file has to be STAGED in this commit —
    the failure this rail exists to catch is a writer whose hook line is on disk but
    not in the commit, so a message that only said "add the line" would produce a
    second refused commit. Column 0 on purpose (#1070), and built in one place so the
    two doors cannot describe the same refusal differently.
    """
    return (f"ORPHAN TOPIC FILE REFUSED: {rec['path']} — this commit ADDS a memory "
            f"topic file that nothing links. No staged line of {MEMORY_REL} ends "
            f"'\u2192 topics/{rec['slug']}', and {LEDGER_REL} carries no "
            f"'**{DISPOSITION_FIELD_LABEL}:**' row naming {rec['path']}. "
            f"Remedy — either one, staged in THIS commit: (a) add the index line to "
            f"{MEMORY_REL} ending '\u2192 topics/{rec['slug']}' and stage that file too "
            f"(git add -- {MEMORY_REL}); the check reads the staged index, so a line "
            f"left dirty in the working tree does not clear it; or (b) write the "
            f"disposition row into {LEDGER_REL} as a line "
            f"- **{DISPOSITION_FIELD_LABEL}:** `{rec['path']}` \u2014 <what happened to the "
            f"file>, which is the remedy for content that is being merged into another "
            f"topic file or retired rather than indexed (#3539's ruling: a retirement "
            f"may not leave a file under {TOPIC_DIR}/ with no index line). "
            f"Nothing is unstaged and nothing is committed (#3545).")


def findings(vault: str | Path) -> tuple[list[dict], list[str], str]:
    """`(records, refusal_lines, why_unread)` — the triple both doors act on.

    Shaped like `ledger_strand_findings.findings` with one more element, and the extra
    element is the whole difference between the two rails: a strand lives in a file
    another commit will fix, so leaving the index alone is right there; an ADD left
    sitting in the index becomes the NEXT commit's add (`git commit` commits the index,
    not the pathspec), and a door that refuses without unstaging has therefore handed
    its own refusal to some later, innocent job's snapshot. So both doors need the
    PATHS, which is what `records` is for, and `--diff-filter=A` is why an untracked
    re-add by their author is the correct end state.

    `why_unread` non-empty means the check did not run, and a door that commits anyway
    has to say so out loud.
    """
    try:
        records = orphan_adds(vault)
        return records, [refusal_text(r) for r in records], ""
    except OrphanTopicCheckUnavailable as exc:
        return [], [], f"orphan topic-file check unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 — a rung that crashes is reported, not fatal
        return [], [], f"orphan topic-file check failed: {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    # `__doc__ or ""` because `python -OO` strips docstrings, and an AttributeError
    # raised here would reach the caller as a bare traceback whose exit code is 1 —
    # the code this tool returns for a REAL refusal, so a stripped build would refuse
    # every vault commit that adds a topic file, quoting no path.
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--vault", default=None,
                    help="vault root (default $LLOYD_VAULT, else ~/obsidian)")
    ap.add_argument("--json", action="store_true",
                    help="print the findings to stdout instead of the refusal block")
    args = ap.parse_args(argv)
    vault = Path(args.vault or (__import__("os").environ.get("LLOYD_VAULT") or
                                vault_root()))
    try:
        records = orphan_adds(vault)
    except OrphanTopicCheckUnavailable as exc:
        print(f"ORPHAN TOPIC FILE CHECK FAILED (not a clean read): {exc}",
              file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"ORPHAN TOPIC FILE CHECK FAILED (not a clean read): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    if not records:
        return 0
    for rec in records:
        print(refusal_text(rec), file=sys.stderr)
    print(f"ORPHAN TOPIC FILE: {len(records)} topic file(s) added under {TOPIC_DIR}/ "
          f"with no staged {MEMORY_REL} line and no {LEDGER_REL} disposition row.",
          file=sys.stderr)
    if not args.json:
        # The contract with `vault-commit.sh`: stdout carries the refused paths and
        # nothing else in this mode (the refusal text is all on stderr), so the door can
        # pull them out of the index. A refusal that leaves a staged ADD in the index is
        # not a refusal — the next `git commit` commits the index, and the next job's
        # snapshot gets refused for a file it never wrote.
        for rec in records:
            print(rec["path"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
