"""Which untracked paths belong to the live checkout, and which appeared during a round.

#1906. A worker's Bash inherits the MCP server's cwd, which is `~/lloyd`, so a
relative write from a review grader or a nightly task lands on production `main`. The
guardian's datawatch notices it and speaks about it hourly, and it cannot say who did
it: the 09-29/30 incident ran eleven `alert` rows over nine hours naming
`/home/alansrobotlab/lloyd/.t`, and the writer was never identified — the fixture
belonged to the grader for round SM_20260930_031934, and only the round's own record
could have said so.

So: snapshot the tree's untracked paths when a round opens, take the tree again after
the round's turn has run, and name what appeared IN THAT ROUND'S RECORD. Detection is
the cheap half; this module exists for the half that makes the alert useless today —
attribution — which is why the answer to "did this round add anything" is never
guessed.

Three states, deliberately not two:

  `["path", ...]`  nothing here was true at the baseline; these are new since
  `[]`              the round added nothing to the live tree
  `None`            UNKNOWN — the baseline is missing or unreadable, so "no strays"
                    cannot be claimed

The third is the whole design. A guard whose missing input reads as a clean answer is
not a guard (the recurring failure class on this board), and the natural way to get
one here is a round that opened before this shipped: its `round_start` row has no
baseline at all, and a diff against "nothing" would report every pre-existing
untracked path — `.t/`, a stale `.patch` — as that round's doing, while a diff written
as `set(after) - set(baseline or [])` would report the same thing and look equally
confident. A baseline of `[]` and a baseline that was never recorded are different
facts and are kept apart by type, not by a sentinel path.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

#: Read-only, and the same call `round.py` already makes for its red-tree check, so
#: a round's two views of the tree are one command with one flag set.
UNTRACKED_CMD = ("git", "-C", "{root}", "status", "--porcelain", "--untracked-files=all")

#: A `git` failure is not an empty tree. `None` means "the tree could not be read",
#[] and every caller must treat that as unknown rather than as clean.
UNREADABLE = None


def untracked(root: Path | str) -> set[str] | None:
    """The paths `git status` calls untracked under `root`, or None if git refused.

    `--untracked-files=all` matters: the default collapses a directory to one line,
    and a fixture written as `.t/05431072c3/r1873/` plus nine files inside it would
    diff as one path, which is one fewer path than anyone can then go and look at.
    Nine entries name nine files, and the one the round actually wrote is among them.

    A non-zero exit returns None rather than raising. This runs inside a round's
    promotion, and a tree that cannot be read must cost a piece of attribution, not
    the promotion — but it must say so, which is what None is for.
    """
    argv = [part.format(root=str(root)) for part in UNTRACKED_CMD]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return UNREADABLE
    if proc.returncode != 0:
        return UNREADABLE
    return parse_porcelain(proc.stdout)


def parse_porcelain(text: str) -> set[str]:
    """The untracked paths out of `git status --porcelain` output.

    Kept separate from the subprocess so the parser is testable on recorded output —
    and so a rename or a quote-escaped path is a parsing question with a text answer,
    not something only reproducible by laying a working tree.

    The two-character status field is the contract: `??` is untracked, everything
    else (` M`, `M `, `A `, `R `, `!!` when ignores are shown) belongs to a
    different question. Splitting on the first space would lose a leading space
    entirely — `" M modified.py"` starts with one — which is how a modified file
    ends up looking untracked.
    """
    out: set[str] = set()
    for line in text.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:]
        if status == "??":
            # A path git had to quote is printed `"with spaces"`; the quotes are
            # git's, not part of the name, and leaving them in produces a path
            # nobody can `ls`.
            out.add(path[1:-1] if path.startswith('"') and path.endswith('"') else path)
    return out


#: Name of the file a round's own uncapped baseline is kept in, inside its round dir.
BASELINE_FILE = "live_baseline.json"


def baseline_path(round_id: str, rounds_dir: Path | str | None = None) -> Path:
    """Where this round's baseline lives: beside its own gate report, in its round dir.

    The baseline goes in the round's record and not only in its `round_start` ledger row
    because that row's sample is capped at twenty paths for legibility, and a subtraction
    against a capped sample would credit the round with every stray past the cap — the
    exact false accusation this module exists to avoid, manufactured by its own logging.
    """
    # A missing state module is raised as OSError on purpose: `write_baseline` catches
    # OSError and returns None, `read_baseline` catches it and returns UNREADABLE, and
    # both are answers this module already has for "no baseline". Letting an ImportError
    # escape would push it into the CALLERS' broad handlers, where it reads as a broken
    # check rather than as a missing record.
    if rounds_dir is None:
        try:
            from scripts.automod import state as S
        except ImportError as exc:  # noqa: BLE001 — see the note above
            raise OSError(f"round state unavailable: {exc}") from exc
        rounds_dir = S.ROUNDS_DIR
    return Path(rounds_dir) / round_id / BASELINE_FILE


def write_baseline(round_id: str, paths: set[str],
                   rounds_dir: Path | str | None = None) -> Path | None:
    """Record the whole untracked set, uncapped, as this round's baseline.

    Returns the path written, or None if it could not be. A None here costs the round
    its attribution — `read_baseline` will then report unknown — and must never be
    papered over with an empty write, which would read as "this round opened on a clean
    tree" and convict it of every stray already in it.
    """
    try:
        path = baseline_path(round_id, rounds_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"round_id": round_id,
                                   "untracked": sorted(paths)}, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)
        return path
    except OSError:
        return None


def read_baseline(round_id: str,
                  rounds_dir: Path | str | None = None) -> set[str] | None:
    """What this round opened with, or None if nobody recorded it.

    None is the same UNREADABLE the tree read returns, and means the same thing: a
    round that opened before this shipped, one whose write failed, and one whose file
    has been swept all have no baseline, and none of them may be answered with `[]`.
    A malformed payload is treated as absent for the same reason — the file being
    there is not evidence of what it said.
    """
    try:
        payload = json.loads(baseline_path(round_id, rounds_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return UNREADABLE
    paths = payload.get("untracked") if isinstance(payload, dict) else None
    if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        return UNREADABLE
    return set(paths)


def appeared(baseline: set[str] | None, now: set[str] | None) -> list[str] | None:
    """The paths that are untracked NOW and were not at the baseline.

    None from either side, and None back: an unreadable baseline or an unreadable
    second look both make "nothing new appeared" an unsupported claim, and both are
    indistinguishable from a clean tree by any caller that did not check. Returning
    None here rather than the empty set is the only thing that keeps a round that
    could not be measured from being reported as a round that wrote nothing.

    Sorted, because this list goes into a ledger row that a person reads, and a set's
    order changes between runs on nothing more than a hash seed.
    """
    if baseline is None or now is None:
        return None
    return sorted(now - baseline)
