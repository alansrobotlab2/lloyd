#!/usr/bin/env python3
"""Warn that a staged skills/ or knowledge/ file has a whole-file witness in the suite.

Item #2381, the recurrence tripwire for #2284. A `live_vault` node can pin a vault
file by WHOLE FILE: `tests/fixtures/step_2a_ter_curation_witness_2212.md` is a
committed copy of `skills/nightly-reflection-knowledge-write/step-2a-ter-curation.md`,
`tests/test_memory_ledger_bound.py::test_the_live_curation_skill_has_not_drifted_from_the_witness_the_ban_reads`
byte-compares the two, and
`::test_the_curation_skill_witness_carries_no_ledger_cap_at_the_topic_ceiling`
asserts that copy is 15,555 B / 74 newlines. Vault `e45a6b6c` appended one bullet to
§2a-ter on 2026-10-06, nobody re-froze the copy, and nothing said so until the next
morning's #83 pre-flight ran the marked node and went red. The job that wrote the
line had no way to know the file it appended to was frozen anywhere.

So the knowing moves to the one door that job commits through.
`scripts/util/vault-commit.sh` runs this after staging and before `git commit`: for
every staged path under `skills/` or `knowledge/`, normalize the basename (drop the
extension, `-` -> `_`, lowercase) and name any `tests/fixtures/*_witness_*.md` whose
name contains that stem. One line per match, with BOTH sides' byte and newline
counts, so the reader can see how far apart the two are without running `wc`.

REPORT ONLY, exactly like `autonomy_status_findings.py` (#1127) and
`skill_path_findings.py` (#1969) before it, for the same two reasons: this wrapper
also commits other writers' state, so a refusal would let one job's drift block
every later job's pre-flight snapshot; and the re-freeze is deliberately not this
job's call to make — the byte pin exists to make a re-freeze a reviewed act, so the
line tells the writer to file a draft, not to edit the fixture. Nothing prints when
nothing matches, the case every ordinary commit takes.

Output goes to STDOUT, never stderr. #1070 makes the `unattributed dirty state:`
path list the LAST thing on stderr so a job can copy its tail verbatim, and
`tests/test_vault_commit_attribution.py` pins that; a line printed to stderr after
that block would become the tail. That is why both earlier rungs print to stdout,
and why this one does.

The fixture corpus is the checkout this script is installed in, so a round's
worktree answers about its own `tests/fixtures`, and a fixture frozen later is
picked up with no change here.

    # at commit time, over the index (what the wrapper does)
    python3 ~/lloyd/scripts/util/witness_fixture_findings.py --repo ~/obsidian
    # a dry run over an explicit path list
    python3 ~/lloyd/scripts/util/witness_fixture_findings.py --repo ~/obsidian \
        --path skills/nightly-reflection-knowledge-write/step-2a-ter-curation.md
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

#: A staged path that can plausibly have a whole-file witness: a skill page or a
#: knowledge note. The convention names the fixture after the file, so only these
#: two segments are worth a lookup; a `memory/` or `backlog/` file that happens to
#: share a basename is not what any witness was frozen from.
_WITNESSED_SEGMENT_RE = re.compile(r"^(?:skills|knowledge)/.+")

#: The committed-witness naming convention: `<stem>_witness_<item>.md`.
_WITNESS_GLOB = "*_witness_*.md"

ADVICE = ("re-freezing it is a reviewed act, not a nightly's — the byte pin in the "
          "node that freezes it is what makes an append visible. File a one-line "
          "backlog draft naming the fixture, both counts above, and the re-freeze "
          "recipe (re-measure `wc -c` and `tr -cd '\\n' | wc -c` from the new file, "
          "and move the byte/newline literals in the pinning node in the same "
          "change). Do not edit the fixture from the committing job, and do not add "
          "an automatic re-generation path. #2284")


def stem_of(rel: str) -> str:
    """The stem a witness fixture would be named after for vault path `rel`.

    Basename, extension dropped, `-` -> `_`, lowercase: the normalization that turns
    `skills/nightly-reflection-knowledge-write/step-2a-ter-curation.md` into
    `step_2a_ter_curation`, which is the leading part of that file's fixture name.
    """
    name = Path(rel).name
    suffix = Path(name).suffix
    if suffix:
        name = name[: -len(suffix)]
    return name.replace("-", "_").lower()


def witness_fixtures(stem: str, fixtures_dir: Path | None = None) -> list[Path]:
    """Every `tests/fixtures/*_witness_*.md` whose name contains `stem`, sorted."""
    if not stem:
        return []
    directory = fixtures_dir or FIXTURES_DIR
    if not directory.is_dir():
        return []
    return sorted((fx for fx in directory.glob(_WITNESS_GLOB)
                   if stem in fx.name.lower()),
                  key=lambda fx: fx.name)


def witness_matches(paths: list[str],
                    fixtures_dir: Path | None = None) -> list[tuple[str, Path]]:
    """(staged path, fixture) for every witnessed path in a staged-path list.

    This is the matcher the acceptance clause names: paths in, pairs out, no git and
    no vault read, so a caller can ask it of a path list it already holds.
    """
    hits: list[tuple[str, Path]] = []
    for rel in paths:
        if not _WITNESSED_SEGMENT_RE.match(rel):
            continue
        for fx in witness_fixtures(stem_of(rel), fixtures_dir):
            hits.append((rel, fx))
    return hits


def counts_of(data: bytes) -> tuple[int, int]:
    """(bytes, newlines) — the two figures `wc -c` and `tr -cd '\\n' | wc -c`."""
    return len(data), data.count(b"\n")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def index_blobs(repo: Path) -> list[tuple[str, bytes]]:
    """(vault-relative path, staged bytes) for every row in `repo`'s index.

    Read off the index and not the worktree because that is what the commit about to
    happen records: a partially-staged file's worktree copy is not its committed
    copy, and the counts have to describe the bytes this commit certifies.
    """
    out = _git(repo, "diff", "--cached", "--name-only", "-z")
    rows: list[tuple[str, bytes]] = []
    for rel in sorted(p for p in out.split("\0") if p):
        proc = subprocess.run(("git", "-C", str(repo), "show", f":{rel}"),
                              capture_output=True, timeout=60)
        if proc.returncode != 0:
            continue               # deleted in the index: nothing staged to compare
        rows.append((rel, proc.stdout))
    return rows


def worktree_blobs(repo: Path, paths: list[str]) -> list[tuple[str, bytes]]:
    """(path, bytes) read from `repo`'s worktree, for a `--path` dry run. A path the
    worktree does not have is skipped rather than reported as 0 B."""
    rows: list[tuple[str, bytes]] = []
    for rel in paths:
        f = repo / rel
        if f.is_file():
            rows.append((rel, f.read_bytes()))
    return rows


def _display(fixture: Path) -> str:
    """Repo-relative inside the checkout, absolute when the corpus came from a
    caller's tmp directory."""
    try:
        return str(fixture.relative_to(REPO_ROOT))
    except ValueError:
        return str(fixture)


def warning_line(rel: str, fixture: Path, fixture_counts: tuple[int, int],
                 side_counts: tuple[int, int]) -> str:
    """The line a committing job reads: the fixture, and both sides' byte and
    newline counts, because a warning that says only "drifted" cannot be triaged
    from the job log that carries it."""
    return (f"witness WARNING: {rel} has a whole-file witness at {_display(fixture)} "
            f"— fixture {fixture_counts[0]} B / {fixture_counts[1]} newlines, staged "
            f"{side_counts[0]} B / {side_counts[1]} newlines. A live_vault node "
            f"byte-compares the two, so that node is red until the copy is "
            f"re-frozen. {ADVICE}")


def report(rows: list[tuple[str, bytes]],
           fixtures_dir: Path | None = None) -> int:
    """Print one warning per witnessed path in `rows`. Always returns 0: this
    reports, it does not block. Silence is the normal answer — an ordinary nightly
    commit stages no witnessed file, and a permanent alarm is a disabled alarm."""
    blobs = dict(rows)
    for rel, fixture in witness_matches(sorted(blobs), fixtures_dir):
        side = blobs.get(rel)
        if side is None or not fixture.is_file():
            continue                                # fixture moved out from under us
        print(warning_line(rel, fixture, counts_of(fixture.read_bytes()),
                           counts_of(side)), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("VAULT_DIR", str(Path.home() / "obsidian")),
                        help="git repo to inspect (default $VAULT_DIR or ~/obsidian)")
    parser.add_argument("--path", action="append", default=None, metavar="REL",
                        help="check this vault-relative path instead of the index "
                             "(repeatable); reads the worktree copy")
    args = parser.parse_args(argv)
    repo = Path(args.repo).expanduser()
    try:
        rows = (worktree_blobs(repo, args.path) if args.path
                else index_blobs(repo))
    except Exception as exc:                        # noqa: BLE001 - never block a commit
        print(f"witness: CHECK FAILED ({exc})", flush=True)
        return 0
    return report(rows)


if __name__ == "__main__":
    sys.exit(main())
