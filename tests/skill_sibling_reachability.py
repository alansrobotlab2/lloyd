"""Does a spilled skill's `SKILL.md` actually point at text that is where it says it is?

`skills/autonomy-data-pipeline/SKILL.md` carries this sentence as its last line:

    Folded 2026-11-22 (#4307): every section above whose text is not here is in the
    sibling named under its heading, and `tests/skill_sibling_reachability.py` holds
    that to be true.

That is a published claim about a mechanism, and until this file existed nothing
checked it: the citation named a path that was not in the checkout, which is the
exact defect `tests/test_skill_tool_names.py::test_no_active_skill_or_task_names_a_path_absent_from_the_checkout`
exists to catch, and it caught it (three nodes red, item #4496). The prose was
written a day ahead of its enforcement — the failure mode `automod-change-own-code`
calls "a citation to an unlanded test".

What is worth holding is not the sentence but the property behind it. A fold moves a
section's *text* into a sibling and leaves `SKILL.md` holding the heading plus a
pointer line, and a run (or a test) that reads only `SKILL.md` then reports a MOVED
rule as a DELETED one — that misreading is what #4190 and #4307 both recorded. The
readers recover it through `tests/skill_folder.py`'s `splice_union`, which matches a
sibling's section to `SKILL.md`'s **by title**. So the one property that decides
whether a fold is safe is:

* the file a pointer names exists in the folder,
* it carries that section's heading **verbatim**, and
* under that heading there is text.

Anything else — a renamed heading, a sibling that was never written, a section that
moved but whose pointer was left behind — makes `splice_union` park the prose in a
tail append (whole-file scans still see it, section-scoped readers do not) or drop
the section's content silently. The two checkers below are pure functions over text
so a node can plant each shape under `tmp_path` and watch it go red; `folder_violations`
is the disk route the live-skill node uses.

Fleet scope is deliberately narrow. `check_skill_siblings` is only asked of a folder
that PUBLISHES the promise (its `SKILL.md` carries a pointer line); the nineteen
sibling-carrying skills measured 2026-11-22 were folded by other rounds under other
contracts, and a check that reds the next author for a prose shape they never agreed to
is a check that gets disabled (the argument `tests/test_skill_tool_names.py` makes for
its own ledger). `unnamed_siblings` is reported by the census node rather than asserted,
for the same reason — 20 of `kg-mention-classifier`'s `run-log-*.md` siblings were named
by nothing in its own `SKILL.md` on that same date.
"""
from __future__ import annotations

import re
from pathlib import Path

from tests.skill_folder import MAIN, sections, splice_union

#: The pointer line a folded section leaves behind: ``Full text: `step2-index-rebuild.md`
#: — read before running this step.`` Only the named filename is captured; the trailing
#: instruction varies per section and is not part of the contract.
POINTER_RE = re.compile(r"Full text: `([^`]+\.md)`")

#: A `- \`post-flight.md\` — what it holds` bullet in the skill's own file index.
INDEX_BULLET_RE = re.compile(r"^- `([^`]+\.md)`", re.M)


def pointer_rows(main: str) -> list[tuple[str, str]]:
    """`(section title, named sibling)` for every pointer line, in document order.

    Keyed by the section the pointer sits in, because that title is the join key
    `splice_union` uses. A pointer parked above the first heading has no section, so it
    is not a row here — that prose belongs to no section and is caught by the sibling's
    own preamble rule, not by this one.
    """
    rows: list[tuple[str, str]] = []
    for title, _line_start, body_start, body_end in sections(main):
        for name in POINTER_RE.findall(main[body_start:body_end]):
            rows.append((title, name))
    return rows


def unreachable(main: str, siblings: dict[str, str]) -> list[str]:
    """Every pointer whose text is NOT reachable in the sibling it names, as messages.

    Three ways to be unreachable, and the message names which:

    * the named file is not in the folder at all;
    * it is, but carries no heading with this section's verbatim title, so
      `splice_union` has nowhere to splice it and a section-scoped reader loses it;
    * it carries the heading but the body under it is empty — the heading is a
      landmark, not the text the sentence promised.
    """
    problems: list[str] = []
    for title, name in pointer_rows(main):
        text = siblings.get(name)
        if text is None:
            problems.append(f"`{name}` named under `{title}` is not in the folder")
            continue
        bodies = {t: (body_start, body_end)
                  for t, _line_start, body_start, body_end in sections(text)}
        if title not in bodies:
            problems.append(
                f"`{name}` named under `{title}` carries no heading with that verbatim "
                f"title (it has: {sorted({t for t, _, _, _ in sections(text)})})")
            continue
        body_start, body_end = bodies[title]
        if not text[body_start:body_end].strip():
            problems.append(f"`{name}` carries `{title}` with no text under it")
    return problems


def missing_index_files(main: str, siblings: dict[str, str]) -> list[str]:
    """Files the skill's own `Files in this skill` index names but the folder lacks."""
    return [name for name in INDEX_BULLET_RE.findall(main) if name not in siblings]


def unrecovered_spans(main: str, union: str,
                      siblings: dict[str, str]) -> list[str]:
    """Pointered sections whose spliced union does NOT carry the sibling's text.

    This is the promise as a reader experiences it: after `splice_union`, the span
    under the section's heading must contain the sibling's body for that heading.
    `unreachable` proves the parts; this proves the composition, and
    `folder_violations` runs both over a folder on disk — which is how the six
    modules that read a folded skill keep their section-scoped assertions.
    """
    problems: list[str] = []
    union_spans: dict[str, tuple[int, int]] = {}
    for title, _ls, body_start, body_end in sections(union):
        union_spans.setdefault(title, (body_start, body_end))
    for title, name in pointer_rows(main):
        text = siblings.get(name)
        span = union_spans.get(title)
        if text is None or span is None:
            continue
        for s_title, _ls, s_body, s_end in sections(text):
            if s_title == title:
                first = next((line.strip() for line in text[s_body:s_end].splitlines()
                              if line.strip()), "")
                if first and first not in union[span[0]:span[1]]:
                    problems.append(
                        f"`{title}`: `{name}`'s first body line is absent from that "
                        f"section's span in the spliced union — it was appended to the "
                        f"tail, so a section-scoped reader never sees it: {first[:70]!r}")
    return problems


def folder_violations(folder: Path, encoding: str = "utf-8") -> list[str]:
    """All three checks over a skill folder read off disk.

    Raises `FileNotFoundError` when the folder has no `SKILL.md`: the caller names the
    folder it meant to police and a missing index is that folder's own failure, not a
    reason to report nothing.
    """
    main = (folder / MAIN).read_text(encoding=encoding)
    siblings = {p.name: p.read_text(encoding=encoding)
                for p in sorted(folder.glob("*.md")) if p.name != MAIN}
    return (unreachable(main, siblings)
            + missing_index_files(main, siblings)
            + unrecovered_spans(main, splice_union(main, sorted(siblings.items())),
                                siblings))


def publishes_reachability_claim(main: str) -> bool:
    """Does this `SKILL.md` hold itself to this contract?

    A fold that left a pointer line is the fold that made the promise; the check is
    asked of those folders and no others (see the module docstring on fleet scope).
    """
    return bool(POINTER_RE.search(main))
