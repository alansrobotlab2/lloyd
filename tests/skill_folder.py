"""Reading one skill *folder* — `SKILL.md` plus the sibling `.md` files a spill moved
sections to — for the four modules that grade `skills/retrieval-eval/SKILL.md` (#4192).

#4190 needs that skill's body under 6,500 chars (`SKILL_BODY_CHAR_CAP = 12_000` at
`scripts/automod/vault_round.py:2874` is the growth refuse; #4151's drafted 5,412-char
booking needs a post-spill body <= 6,588), and the only way to get 62,365 chars there is
to move whole sections into siblings under the same directory. Four modules assert that
skill's prose, and until today each read `SKILL.md` alone — which is why #4192 was a
blocker and not a difficulty: the prose those nodes grade had nowhere to go.

The widening keeps the two protections each reader was written for:

* **Section scope.** `tests/test_retrieval_eval_skill_ci_contract.py`'s `section()`
  docstring says why a whole-file scan is not enough: "a keyword scanned across the whole
  file would pass on a mention in `## Notes` while Step 3 said something else, and Step 3
  is where the verdict gets authored". So the union is built by *splicing each sibling's
  section body in beneath the heading it shares*, never by concatenating the files: the
  span under a heading grows to include the sibling's body for that heading, and a keyword
  parked under a different heading — in `SKILL.md` or in a sibling — still lands in a
  different span. Section *existence* stays `SKILL.md`'s alone (`section()` asserts it
  before it looks at the union), so a heading that exists only in a sibling is still the
  `no section headed …` failure.
* **Committed HEAD.** `tests/test_eval_trend_stats.py` and
  `tests/test_eval_label_agreement.py` read the vault through git precisely so an
  uncommitted working-tree edit cannot make a landed contract read green. Their sibling
  enumeration therefore comes from `git ls-tree --name-only HEAD <dir>` and each body
  from `git show HEAD:<sibling>`: a sibling present in the working tree but absent from
  HEAD contributes nothing, and an unreadable HEAD still `pytest.fail`s rather than
  skipping. The two disk-route modules keep the semantics they already had — they read
  `SKILL.md` off disk, and they read its siblings off disk.

`splice_union` is pure text, which is what lets a node build a folder under `tmp_path`,
point `LLOYD_VAULT` at it, and watch both halves of that contract fail or pass.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

#: The file whose headings define the skill's structure. Existence of a section is a
#: claim about this file only; its *content* may live in a sibling.
MAIN = "SKILL.md"

_HEADING = re.compile(r"^(#{2,3}) (.*)$", re.MULTILINE)


def sections(text: str) -> list[tuple[str, int, int, int]]:
    """`(title, heading-line start, body start, body end)` per H2/H3 heading, in order.

    The body of a section runs to the next heading of either level, which is the same cut
    `section()` makes, so what this returns and what a caller grades cannot disagree.
    Titles are compared stripped of surrounding whitespace but otherwise verbatim: a
    spilled section keeps its heading line, so the title is the key that survives a move.
    """
    marks = [(m.start(), m.end(), m.group(2).strip()) for m in _HEADING.finditer(text)]
    return [(title, line_start, body_start,
             marks[index + 1][0] if index + 1 < len(marks) else len(text))
            for index, (line_start, body_start, title) in enumerate(marks)]


def splice_union(main: str, siblings: list[tuple[str, str]]) -> str:
    """`main` with every sibling's section body spliced in beneath its shared heading.

    Splicing rather than concatenating keeps the result section-scoped: a span under
    `## Step 3` in the union contains `main`'s own `## Step 3` text plus each sibling's
    `## Step 3` text and nothing else, so the keyword tests keep their teeth. Slices a
    caller takes between two headings (the two `TEXT[TEXT.index(...) : …]` reads in
    `tests/test_retrieval_eval_skill_ci_contract.py`) keep working too, because a
    sibling's prose arrives inside the region its heading belongs to instead of at the
    end of the file.

    A sibling section whose heading `main` does not carry has nowhere to be spliced, so it
    is appended whole — heading line included — after `main`. Whole-file assertions still
    see that prose; `section()` will not, because it refuses a heading `main` lacks. That
    is clause 3's asymmetry, not an oversight.

    Insertions apply at descending offsets so each offset stays valid while they are being
    made, and siblings sharing one heading are laid down in the order they were passed.
    """
    end_of_main_span: dict[str, int] = {}
    for title, _line_start, _body_start, body_end in sections(main):
        end_of_main_span.setdefault(title, body_end)

    inserts: list[tuple[int, int, str]] = []
    tail: list[str] = []
    for sibling_index, (_name, text) in enumerate(siblings):
        own = sections(text)
        # `own[0][1]` is the first heading's line start; everything before it is prose that
        # belongs to no heading in this file, so it has nowhere to be spliced and joins the
        # tail. `own[0][0]` is the title — a string — and slicing by it is a TypeError.
        preamble_end = own[0][1] if own else len(text)
        if text[:preamble_end].strip():
            tail.append(text[:preamble_end])
        for section_index, (title, line_start, body_start, body_end) in enumerate(own):
            if title in end_of_main_span and text[body_start:body_end].strip():
                inserts.append((end_of_main_span[title],
                                sibling_index * 1000 + section_index,
                                text[body_start:body_end]))
            else:
                tail.append(text[line_start:body_end])

    out = main
    for offset, _order, chunk in sorted(inserts, key=lambda row: (-row[0], -row[1])):
        out = out[:offset] + chunk + out[offset:]
    if tail:
        out = out.rstrip("\n") + "\n" + "\n".join(tail)
    return out


def _read(path: Path, encoding: str | None) -> str:
    # `encoding=None` is pathlib's own "use the locale default", which is what the
    # disk-route modules did before this file existed; pass it through rather than
    # silently changing one module's decoding.
    return path.read_text(encoding=encoding)


def disk_skill_folder(folder: Path,
                      encoding: str | None = None) -> tuple[str, str]:
    """`(SKILL.md's text, the union)` for a skill folder read off disk.

    Raises `FileNotFoundError` when the folder has no `SKILL.md`; the two disk-route
    modules guard that with their existing existence check.
    """
    main = _read(folder / MAIN, encoding)
    siblings = sorted((path.name, _read(path, encoding))
                      for path in folder.glob("*.md") if path.name != MAIN)
    return main, splice_union(main, siblings)


def git_skill_folder(vault: Path, rel_dir: str,
                     timeout: int = 60) -> tuple[str, str, str | None]:
    """`(SKILL.md's text, the union, error)` taking every file from HEAD, not the worktree.

    `rel_dir` is the folder's path inside the vault, e.g. `skills/retrieval-eval`. Returns
    an error string instead of raising so each caller keeps its own `pytest.fail` wording:
    these two fixtures fail loudly because a skipped acceptance check is the gate that
    reads green while nobody looks, which is the argument their own comments make.
    """
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(vault), *args],
                              capture_output=True, text=True, timeout=timeout)

    listing = git("ls-tree", "--name-only", "HEAD", rel_dir.rstrip("/") + "/")
    if listing.returncode != 0:
        return "", "", (f"`git -C {vault} ls-tree --name-only HEAD {rel_dir}/` failed "
                        f"({listing.stderr.strip()[:200]})")
    paths = sorted(p for p in listing.stdout.split() if p.endswith(".md"))
    main_path = f"{rel_dir.rstrip('/')}/{MAIN}"
    if main_path not in paths:
        return "", "", (f"`git -C {vault} ls-tree --name-only HEAD {rel_dir}/` listed no "
                        f"{main_path} (listed: {paths})")

    bodies: dict[str, str] = {}
    for path in paths:
        shown = git("show", f"HEAD:{path}")
        if shown.returncode != 0:
            return "", "", (f"`git -C {vault} show HEAD:{path}` failed "
                            f"({shown.stderr.strip()[:200]})")
        bodies[path] = shown.stdout

    main = bodies[main_path]
    siblings = [(path.rsplit("/", 1)[1], bodies[path])
                for path in paths if path != main_path]
    return main, splice_union(main, siblings), None
