#!/usr/bin/env python3
"""okf_stranded.py — the second frontmatter block stranded in a document BODY.

A July 2026 wrapper pass stamped a frontmatter block onto vault markdown without
deleting the original one. The leading block parses, so every strict reader is
satisfied and the real block below it is invisible: `validate_okf.py` called
those files conformant and `okf_migrate.py --repair-only` called them unchanged
for five weeks (item #960). This module is the one detector both scripts import,
so the two cannot disagree about what the family is — and neither keeps its own
copy of how big it is.

Two shapes exist on disk, and both are found:

  fenced     the body holds ``---\\nkey: v\\n---`` — the original block kept its
             fences and sits under the stamp (the digest/updates files).
  unfenced   the stamp's closing ``---`` WAS the original block's opening fence,
             so the body simply begins with a run of ``key:`` lines
             (``knowledge/foundational/thinking-foundations.md`` line 10).

Two shapes are deliberately NOT reported, because a ``key:`` line in prose is not
a frontmatter block:

  * anything inside a ``` fenced code block — the schema page's own example
    (``knowledge/KNOWLEDGE_SCHEMA.md``) is the live instance;
  * a lone ``key:`` line with prose above and below it — the shape of the five
    ``type: user-facing description`` lines inside an HTML comment in
    ``knowledge/tools/openclaw/prs.md``.

Strandedness is judged STRUCTURALLY, never by parsing the block: #961's glued
flow-list corruption (`tags: [a,b,cdate: 2025-07-09`) makes a stranded block
unparseable as YAML, so a parse gate would blind this detector to a third of the
family it exists to find. Values are parsed by whoever needs them, and the two
sites are deliberately different: the migrator's LEADING block goes through its
existing recovery chain (`okf_migrate.load_fm`, which folds orphaned tag items and
re-quotes a glued fence), while a BODY block is parsed plainly in
`okf_migrate.collapse_stranded_frontmatter` and the merge is REFUSED if it will not
parse. Values a recovery chain guessed its way into are acceptable in a block that
is already broken and being rewritten wholesale; they are not acceptable merged into
the one block that currently parses.

The known legacy set is a frozen, checked-in allow-list (``okf_stranded_known.txt``,
one vault-relative path per line). A stranded block ON the list is counted as
`known-stranded`; one NOT on it is a new violation. That is the regression guard
the family has never had, and it is the single source of truth the write-side
vocabulary guard defers to (#960 clause 5) instead of carrying a count of its own.
When #478's data fix lands, regenerate the list with
``python scripts/vault/okf_stranded.py --write-allowlist`` — it should come back
empty, and the validator's known-stranded count goes to 0 with it.

Usage:
    python scripts/vault/okf_stranded.py [--dir NAME] [--root PATH]
                                         [--write-allowlist]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.paths import VAULT_ROOT  # noqa: E402

# Same skips as `iter_md` in validate_okf.py and okf_migrate.py — the three
# scripts must walk the same tree or their counts mean different things.
EXCLUDE_DIRS = {"templates", "images", ".git", ".obsidian", ".trash"}
EXCLUDE_FILES = {"tags.md", "index.md", "log.md"}

ALLOWLIST_PATH = Path(__file__).with_name("okf_stranded_known.txt")

# A top-level YAML key line. Deliberately not indented: an indented `type:` is
# inside some nested mapping and is not the page's type.
_KEY_RE = re.compile(r"^([A-Za-z0-9_-]+):(?:[ \t]|\r?$)")
_CONT_RE = re.compile(r"^[ \t]+\S")
_FENCE_LINE = re.compile(r"^---[ \t]*$")
_CODE_FENCE_RE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})(?P<info>[^\n]*)$")

# How far past an opening `---` to look for the block's keys and closing fence.
_MAX_BLOCK_LINES = 200


class Stranded(NamedTuple):
    """One frontmatter block found sitting in a document body."""

    line: int      # 1-based line the block starts on (its `---`, or its first key)
    keys: tuple    # top-level keys it carries, in document order
    text: str      # the raw block, its own fences included when it has them
    inner: str     # just the key lines, for a caller that wants to parse values
    start: int     # char offset of `text` in the document
    end: int       # char offset just past `text` (and its trailing newline)
    fenced: bool   # whether `text` opens with its own `---` fence

    @property
    def has_type(self) -> bool:
        """Whether this block carries a `type:` — the contradiction #478 counts."""
        return "type" in self.keys


def _top_level_keys(lines: Iterable[str]) -> tuple:
    return tuple(m.group(1) for m in
                 (_KEY_RE.match(ln) for ln in lines) if m)


def _leading_block_end(text: str) -> int | None:
    """Line index (0-based) the body starts on, or None if there is no leading
    ``---`` block at all. Whether the leading block EXISTS and parses is
    validate_okf's business; this module only answers "what is stranded below
    it", and a file with no leading block has nothing stranded by one."""
    lines = text.split("\n")
    if not lines or not _FENCE_LINE.match(lines[0]):
        return None
    for i in range(1, len(lines)):
        if _FENCE_LINE.match(lines[i]):
            return i + 1
    return None


def _code_fenced_lines(lines: list[str]) -> set[int]:
    """Line indices inside a ``` / ~~~ code block, fence lines included.

    Only a fence that CLOSES makes a region. An unterminated opener is prose:
    treating it as a fence inverts the parity of everything below it and blinds
    the scanner downstream — during development that made a fenced example inside
    a research doc read as a stranded block three hundred lines later. A closing
    marker takes the same character and is at least as long as the opener, so a
    four-tick example block can quote a three-tick one.
    """
    covered: set[int] = set()
    i = 0
    while i < len(lines):
        mo = _CODE_FENCE_RE.match(lines[i])
        if not mo:
            i += 1
            continue
        marker = mo.group("marker")
        close = None
        for k in range(i + 1, len(lines)):
            co = _CODE_FENCE_RE.match(lines[k])
            if (co and co.group("marker")[0] == marker[0]
                    and len(co.group("marker")) >= len(marker)
                    and not co.group("info").strip()):
                close = k
                break
        if close is None:
            i += 1                     # never closed: not a region, keep scanning
            continue
        covered.update(range(i, close + 1))
        i = close + 1
    return covered


def _scan_run(lines: list[str], j: int) -> tuple[int, tuple]:
    """Extend a key run from line j: consecutive top-level key lines and their
    indented continuations. Returns (index one past the run, key names)."""
    keys: list[str] = []
    k = j
    while k < len(lines):
        mo = _KEY_RE.match(lines[k])
        if mo:
            keys.append(mo.group(1))
            k += 1
            continue
        if _CONT_RE.match(lines[k]):
            k += 1
            continue
        break
    return k, tuple(keys)


class _Span(NamedTuple):
    """Where a candidate block sits. `first` is the block's first line (its
    opening `---` when it has one, else its first key line); the key lines are
    ``lines[key_first:key_last]``; `closed` is the closing `---` or None."""

    first: int
    key_first: int
    key_last: int      # exclusive
    closed: int | None


def _candidates(lines: list[str], body_start: int,
                code: set[int]) -> Iterable[_Span]:
    """Yield each candidate block in the body, skipping code-fenced regions.

    Two rules, both structural:

      deeper in the body, a ``---``-delimited block holding at least one key
      line — the fences make a one-key fragment (the ``---\\nsegment: backlog\\n---``
      shape) unambiguous on its own;

      at the very start of the body only, a run of at least two key lines, with
      an opening ``---`` allowed but not required — the opening fence is often
      unclosed there (``knowledge/stack-updates/2026-05-16-vllm.md``) and the
      wrapper pass sometimes left no fence at all.

    The two-key floor on the second rule is what keeps a single ``type:`` line
    quoted in prose from reading as a block.
    """
    j = body_start
    while j < len(lines):
        if j in code:
            j += 1
            continue

        at_body_start = all(not lines[x].strip() for x in range(body_start, j))
        if at_body_start:
            opened = bool(_FENCE_LINE.match(lines[j]))
            key_first = j + 1 if opened else j
            end, keys = _scan_run(lines, key_first)
            if len(keys) >= 2:
                # The run often ends on its own `---` — the stamp swallowed only
                # the OPENING fence, or none at all. That fence belongs to the
                # block: leave it behind and the repair collapses two blocks into
                # one block plus a stray `---`, which is still a delimiter
                # boundary for every lenient parser downstream.
                trailing = (end if end < len(lines)
                            and _FENCE_LINE.match(lines[end]) else None)
                yield _Span(first=j, key_first=key_first, key_last=end,
                            closed=trailing)
                j = (end + 1) if trailing is not None else max(end, j + 1)
                continue

        if _FENCE_LINE.match(lines[j]):
            k = j + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            end, keys = _scan_run(lines, k)
            closed = None
            probe = end
            while probe < min(len(lines), j + _MAX_BLOCK_LINES):
                if _FENCE_LINE.match(lines[probe]):
                    closed = probe
                    break
                if lines[probe].strip():
                    break                       # prose resumes: not a block
                probe += 1
            # `closed >= end` because the key run stops AT its closing fence:
            # `---\nkey: v\n---` has the fence on the very line the run ends on.
            if keys and closed is not None and closed >= end:
                yield _Span(first=j, key_first=k, key_last=end, closed=closed)
                j = closed + 1
                continue

        j += 1


def find_stranded_frontmatter(text: str) -> list[Stranded]:
    """Every frontmatter block stranded in `text`'s body, in document order.

    An empty list is the normal answer, and is also the answer for a document
    whose only `key:` lines are quoted in prose or inside a ``` block.
    """
    body_start = _leading_block_end(text)
    if body_start is None:
        return []
    lines = text.split("\n")
    offsets: list[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    code = _code_fenced_lines(lines)

    out: list[Stranded] = []
    for span in _candidates(lines, body_start, code):
        keys = _top_level_keys(lines[span.key_first:span.key_last])
        if not keys:
            continue
        # The block's last physical line: its closing `---` when it has one,
        # otherwise the last key line of the run.
        last_line = span.closed if span.closed is not None else span.key_last - 1
        start_char = offsets[span.first]
        end_char = offsets[last_line] + len(lines[last_line])
        if end_char < len(text) and text[end_char] == "\n":
            end_char += 1
        out.append(Stranded(line=span.first + 1,
                            keys=keys,
                            text=text[start_char:end_char],
                            inner="\n".join(lines[span.key_first:span.key_last]),
                            start=start_char,
                            end=end_char,
                            fenced=span.first != span.key_first))
    return out


def first_stranded_line(text: str) -> int | None:
    """The 1-based line the first stranded block starts on, or None."""
    found = find_stranded_frontmatter(text)
    return found[0].line if found else None


# ── the frozen legacy set ────────────────────────────────────────────────────

def iter_concept_md(root: Path, only_dir: str | None = None) -> Iterable[Path]:
    """The concept-document walk shared by the three scripts (see EXCLUDE_*)."""
    base = root / only_dir if only_dir else root
    for p in sorted(base.rglob("*.md")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.name in EXCLUDE_FILES or p.name.startswith("_"):
            continue
        yield p


def load_allowlist(path: Path | None = None) -> frozenset[str]:
    """The checked-in known-stranded set, as vault-relative posix paths.

    Resolved through the module global `ALLOWLIST_PATH` at CALL time, so a test
    can aim it at a fixture list without touching the shipped one. A missing file
    is an empty set — a fresh tree has none, which makes the guard stricter,
    never silent.
    """
    p = ALLOWLIST_PATH if path is None else path
    if not p.is_file():
        return frozenset()
    return frozenset(
        ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    )


def scan(root: Path, only_dir: str | None = None) -> dict[str, int]:
    """relpath -> first stranded line, for every concept file that has one."""
    out: dict[str, int] = {}
    for p in iter_concept_md(root, only_dir):
        try:
            line = first_stranded_line(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
        if line:
            out[p.relative_to(root).as_posix()] = line
    return out


def allowlist_disagreements(root: Path, only_dir: str | None = None,
                            allowlist: frozenset[str] | None = None):
    """Where the shipped allow-list and the live tree part company.

    Returns ``(new, resolved)``. `new` are stranded files the list does not cover
    — genuine regressions, which `validate_okf.py` fails on. `resolved` are
    listed files the detector no longer flags, which means #478's data fix
    reached them and the list needs regenerating. Neither side can be read off
    the other: a check that only reads the list misses a new corruption, and one
    that only reads the tree never notices the fix.
    """
    known = load_allowlist() if allowlist is None else allowlist
    found = scan(root, only_dir)
    new = sorted(set(found) - set(known))
    resolved = sorted(set(known) - set(found))
    return new, resolved


def write_allowlist(root: Path, only_dir: str | None = None,
                    path: Path | None = None) -> int:
    """Rewrite the allow-list from the scanned tree. Returns how many files it names."""
    found = scan(root, only_dir)
    target = ALLOWLIST_PATH if path is None else path
    target.write_text(
        "# Known-stranded frontmatter: the legacy set left by the July 2026\n"
        "# wrapper pass (#478's data half). validate_okf.py counts these as\n"
        "# `known-stranded` instead of VIOLATIONS, so the weekly conformance job\n"
        "# stays green while the data is unfixed; anything stranded and NOT\n"
        "# listed here is a violation. okf_taxonomy.py defers to this same file.\n"
        "# Regenerate: python scripts/vault/okf_stranded.py --write-allowlist\n",
        encoding="utf-8")
    with target.open("a", encoding="utf-8") as fh:
        for rel in sorted(found):
            fh.write(f"{rel}\n")
    return len(found)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=None,
                    help="scope the scan to one top-level dir (e.g. knowledge)")
    ap.add_argument("--root", default=None,
                    help="tree to scan instead of app.paths.VAULT_ROOT")
    ap.add_argument("--write-allowlist", action="store_true",
                    help="rewrite okf_stranded_known.txt from the scanned tree")
    args = ap.parse_args()
    root = Path(args.root) if args.root else VAULT_ROOT

    if args.write_allowlist:
        n = write_allowlist(root, args.dir)
        print(f"[okf_stranded] wrote {n} known-stranded paths to {ALLOWLIST_PATH}")
        return 0

    found = scan(root, args.dir)
    known = load_allowlist()
    new, resolved = allowlist_disagreements(root, args.dir, known)
    print(f"[okf_stranded] {len(found)} stranded"
          + (f" in {args.dir}" if args.dir else "") + f", {len(known)} listed")
    if new:
        print(f"  NEW (stranded, not on the allow-list): {len(new)}")
        for r in new[:20]:
            print(f"    {r}:{found[r]}")
    if resolved:
        print(f"  RESOLVED (listed, no longer stranded — regenerate): {len(resolved)}")
        for r in resolved[:20]:
            print(f"    {r}")
    if not new and not resolved:
        print("  allow-list and tree agree")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
