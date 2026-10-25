#!/usr/bin/env python3
"""The one definition of a run-record citation, and whether the record still exists.

A skill page cites an autonomy run record as the evidence for a rule: "on 2026-10-23
this job wrote `run_40_20261023_020003` …". The corpus those citations point into is
deleted on a horizon BY DESIGN — `RUN_RECORD_MAX_AGE_DAYS = 30`
(`scripts/groundskeeper/retention-sweep.py:590`), consumed by `sweep_autonomy_runs`
(same file, `max_age = RUN_RECORD_MAX_AGE_DAYS * 86400`, which unlinks
`autonomy-runs/*/*.md`), run daily by autonomy task #79. So every citation has an
expiry that nothing tells the writer about, and the decay is invisible to CI: the
existing rail (`scripts/util/skill_path_findings.py`) reports a path a skill NEWLY
names while the checkout lacks it, and `tests/test_skill_tool_names.py:417` exempts
`autonomy-runs/` outright via `RUNTIME_FIRST` while its harvest pattern `_LLOYD_PATH`
(:430) only spells paths under `~/lloyd/` — so these paths were never even candidates.
A citation that resolves when it is written and is swept 30 days later never re-trips
any check, because the deletion is the event and no rung is invoked by a deletion.

Measured on 2026-10-25 (#3299) by this module's own extractor, over
`~/obsidian/skills/*/SKILL.md`: **42 (file, id) citations of 40 distinct ids in 13 files;
25 ids resolve, 15 are gone** — 17 file-id pairs spread over 10 files. The item's triage
quoted 51 citations in 13 files with 17 ids gone; both figures are right for the surface
they measured. Reading the WHOLE file text (what that pass did) finds 51 occurrences of 43
ids and 16 gone today, and the gap is exactly this extractor's two rules: 3 occurrences of
3 distinct ids sit inside fenced code blocks, in 2 files (`autonomy-data-pipeline`'s
`run_24_20260930_052951` and `…_094319`, `nightly-skill-consolidation`'s
`run_58_20260916_092545` — the last one of those is itself a dead id, which is why the
whole-text reading counts 16 and the prose reading 15), and the remaining 6 of the 51 are a
second and third mention of an id already cited in the same file, which the extractor
counts once because the land door refuses a citation, not a repetition. The oldest record
day on disk was `20260924` — the sweep running to its own constant.

Scope this module does NOT have, stated so nobody reads the gate as wider than it is:

  * It reads skill bodies only. The same decay reaches tracked code —
    `scripts/consolidation_source_gate.py:24` cites `run_58_20260920_104452` in a
    module docstring, one of the swept ids — and no rule here covers it.
  * It never repairs a citation. The corpus is open-set (the fleet adds citations
    nightly), so a hand-maintained allowlist cannot close a property that must hold
    over it; that is the 2026-10-01 `KNOWN_GOOD_TOPLEVEL` shape. Repairing the 17 is
    worth doing once, and is not this instrument.
  * It refuses only what a diff ADDS. The 10 files carrying dead ids are rewritten
    nightly by `nightly-reflection-*` and `morning-brief-and-triage`; refusing a
    touched skill that merely CARRIES a dead id would block essentially every landing
    that touches them, which is the permanent-alarm-that-becomes-a-disabled-alarm trap
    `scripts/util/skill_path_findings.py:28-31` states. Pre-existing ids are reported,
    never refused.

Resolution is by NAME PREFIX, not exact name, for a reason measured rather than
invented: the writer sometimes appends a 6-hex suffix the citation omits, so
`run_scheduled-task_20261016_233600` is cited while the record on disk is
`run_scheduled-task_20261016_233600_69ae2b.md`. Exact-name matching would report that
as decay it does not have. Prefix matching is sound here because the token already
carries its task id and a second-resolution stamp, so the only longer name it can
prefix is the same record with a suffix.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

#: A run-record citation: `run_<task>_<yyyymmdd>_<hhmmss>`, bare, path-prefixed, with
#: or without `.md`. The task part is lazy and cannot start with a separator, so
#: `run_101_20261022_080154` yields task `101` and
#: `run_scheduled-task_20261016_233600` yields `scheduled-task` rather than swallowing
#: the stamp. Two forms return 0 against a NON-EMPTY corpus and read as "the corpus is
#: clean", both measured on #3299's triage pass: `2026[0-9]{6}` for the date (the date
#: is 8 digits, so that pattern asks for 4 more after `2026` and matches nothing), and
#: a task class written `[A-Za-z0-9.-]` without the underscore, which cannot cross a
#: task id at all. The lookbehind keeps `xrun_5_…` inside a word from counting, while
#: still matching the path-prefixed spelling after its `/`.
CITATION = re.compile(r"(?<![\w.-])run_([A-Za-z0-9][A-Za-z0-9._-]*?)_([0-9]{8})_([0-9]{6})")

#: Opens or closes a fenced block. `~~~` is included because CommonMark allows it, and
#: an untracked fence turns every following line into prose — which is how a citation
#: inside a heredoc example would become a refusal nobody wrote. Same rule as
#: `scripts/skill_lint.py:719-720` for its command sweep.
_FENCE = re.compile(r"^ {0,3}(```|~~~)")

RESOLVES = "resolves"
GONE = "gone"

#: The horizon a citation inherits, quoted from the sweeper rather than copied as a
#: second constant. Read at call time by `horizon_days()`; the number lives in the
#: retention sweep that acts on it.
_SWEEP = Path(__file__).resolve().parent / "groundskeeper" / "retention-sweep.py"


def horizon_days() -> int | None:
    """`RUN_RECORD_MAX_AGE_DAYS` as the sweep spells it, or None if unreadable.

    Parsed from the sweeper's source because importing it runs its module body, and a
    citation-check module that imports a deleter is a module whose import can fail for
    reasons the citation check should not inherit. A refusal that could not read the
    number still refuses — it just says "on the sweep's horizon" — so this is never a
    way to skip the check.
    """
    try:
        text = _SWEEP.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"^RUN_RECORD_MAX_AGE_DAYS\s*=\s*([0-9]+)", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def autonomy_runs_dir() -> Path:
    """Where the records live: `${LLOYD_DATA:-~/lloyd-data}/autonomy-runs`.

    The same rule the sweeper that deletes them uses
    (`scripts/groundskeeper/retention-sweep.py:493-502`, `${LLOYD_DATA:-~/lloyd-data}`
    resolved through `app.data_root`), restated here without the import for the reason
    above. Honouring the variable is what lets a test point the instrument at a fixture
    corpus instead of the live one, which is what makes the branch testable at all.
    """
    env = os.environ.get("LLOYD_DATA")
    root = Path(env) if env else Path.home() / "lloyd-data"
    return root / "autonomy-runs"


def prose(text: str) -> str:
    """`text` with every fenced block removed, so only prose is graded.

    A fenced block is a command a writer shows a reader, and a command that names a
    record is usually `ls`ing it, not citing it as evidence. Refusing the second for
    the first would push writers to spell run ids without backticks, which is worse.
    """
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if m and fence is None:
            fence = m.group(1)
            continue
        if m and m.group(1) == fence:
            fence = None
            continue
        if fence is not None:
            continue
        out.append(line)
    return "\n".join(out)


def citations_in_prose(text: str) -> list[str]:
    """Every run-record citation in `text` outside fences, in first-seen order, deduped.

    The token is the whole `run_<task>_<stamp>` match, so a `.md` suffix, a `~/lloyd-data/`
    prefix and the `autonomy-runs/<task>/` directory are all stripped by construction:
    they are spellings of one citation, not different citations.
    """
    seen: list[str] = []
    for m in CITATION.finditer(prose(text)):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    return seen


def record_names(runs_dir: Path) -> list[str]:
    """Record names in `runs_dir`, with `.md` / `.md.gz` removed, sorted.

    Reads `*/*.md` and `*/*.md.gz` — the sweeper's own glob plus its gzip form — and
    keeps names starting `run_`, so a `wiki-sweep-latest.json` or a legacy
    epoch-millisecond record (`1774837994780.md`, which the sweep also deletes) cannot
    satisfy a citation. A missing directory is an empty list: the corpus being absent is
    what a sweep does to it, and the caller decides whether that is an error.
    """
    if not runs_dir.is_dir():
        return []
    names: list[str] = []
    for path in list(runs_dir.glob("*/*.md")) + list(runs_dir.glob("*/*.md.gz")):
        name = path.name
        for suffix in (".md.gz", ".md"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        if name.startswith("run_"):
            names.append(name)
    return sorted(names)


def verdict(token: str, names: list[str] | set[str]) -> str:
    """`resolves` when a record still exists for `token`, `gone` when none does.

    Exact name, or the name plus a `_`-separated suffix — see the module docstring for
    why prefix matching is the sound reading and not a looseness.
    """
    for name in names:
        if name == token or name.startswith(token + "_"):
            return RESOLVES
    return GONE


def verdicts(text: str, runs_dir: Path | None = None) -> list[tuple[str, str]]:
    """`[(citation, resolves|gone)]` for every citation a body carries."""
    names = record_names(autonomy_runs_dir() if runs_dir is None else runs_dir)
    return [(tok, verdict(tok, names)) for tok in citations_in_prose(text)]


def added_citations(new_text: str, old_text: str) -> list[str]:
    """Citations `new_text` names that `old_text` did not — the added-only set.

    A set difference on the extracted token, not a diff of the files: what the land
    door must refuse is a citation the batch BRINGS, and a reflowed paragraph keeps
    every token it already had, so a line-level diff would report a moved citation as
    a new one and block a writer for re-wrapping a sentence. `old_text` is what HEAD
    holds, which is the empty string for a skill that does not exist yet — a new file
    adds everything it cites, which is right.
    """
    return sorted(set(citations_in_prose(new_text)) - set(citations_in_prose(old_text)))


def gone_citations(tokens: list[str], runs_dir: Path | None = None) -> list[str]:
    """The subset of `tokens` no record answers to. See `verdict` for the rule."""
    names = record_names(autonomy_runs_dir() if runs_dir is None else runs_dir)
    return sorted({t for t in tokens if verdict(t, names) == "gone"})


def skill_files(vault: Path) -> list[Path]:
    """The corpus the report grades: every `skills/**/SKILL.md` under `vault`."""
    return sorted((vault / "skills").glob("**/SKILL.md"))


def report(vault: Path, runs_dir: Path | None = None) -> dict:
    """Counts over the live skill corpus, so the pre-existing dead ids are visible.

    This is ask 2 of #3299: the refusal only ever sees what a diff adds, so without a
    report the 17 ids already dangling are known to nobody and can never be closed.
    """
    corpus = record_names(autonomy_runs_dir() if runs_dir is None else runs_dir)
    cited: list[str] = []
    files: dict[str, list[str]] = {}
    for path in skill_files(vault):
        toks = [t for t in citations_in_prose(path.read_text(encoding="utf-8", errors="replace"))]
        dead = [t for t in toks if verdict(t, corpus) == GONE]
        if dead:
            files[path.relative_to(vault).as_posix()] = dead
        cited.extend(toks)
    gone = sorted({t for toks in files.values() for t in toks})
    return {
        "files_scanned": len(skill_files(vault)),
        "citations": len(cited),
        "distinct": len(set(cited)),
        "gone_distinct": len(gone),
        "gone_citations": sum(len(v) for v in files.values()),
        "gone_files": len(files),
        "by_file": files,
        "gone_ids": gone,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vault", default=str(Path.home() / "obsidian"))
    ap.add_argument("--runs", default=None, help="override the autonomy-runs corpus")
    ap.add_argument("--ids", action="store_true", help="list the dangling ids, not just counts")
    args = ap.parse_args(argv)
    runs = Path(args.runs) if args.runs else None
    r = report(Path(args.vault), runs)
    horizon = horizon_days()
    print(f"run-record citations: {r['citations']} in {r['files_scanned']} skill files "
          f"({r['distinct']} distinct ids)")
    print(f"dangling: {r['gone_distinct']} ids / {r['gone_citations']} citations "
          f"in {r['gone_files']} files")
    print(f"horizon: {horizon if horizon is not None else 'unread'} days "
          f"(RUN_RECORD_MAX_AGE_DAYS) — a citation older than that is expected to be gone")
    if args.ids:
        for name, dead in sorted(r["by_file"].items()):
            print(f"  {name}: {', '.join(dead)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
