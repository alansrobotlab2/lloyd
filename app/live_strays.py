"""Which paths git is not carrying — untracked or ignored — belong to the live checkout,
and which of them appeared during a round.

#1906. A worker's Bash inherits the MCP server's cwd, which is `~/lloyd`, so a
relative write from a review grader or a nightly task lands on production `main`. The
guardian's datawatch notices it and speaks about it hourly, and it cannot say who did
it: the 09-29/30 incident ran eleven `alert` rows over nine hours naming
`/home/alansrobotlab/lloyd/.t`, and the writer was never identified — the fixture
belonged to the grader for round SM_20260930_031934, and only the round's own record
could have said so.

So: snapshot the tree's stray paths when a round opens, take the tree again after
the round's turn has run, and name what appeared IN THAT ROUND'S RECORD. Detection is
the cheap half; this module exists for the half that makes the alert useless today —
attribution — which is why the answer to "did this round add anything" is never
guessed.

"Untracked" here means what `git status` reports as `??` AND as `!!` (#2059). The first
question this module was asked was the wrong one: `--untracked-files=all` on its own
lists untracked-and-not-ignored paths, and every runtime store `datawatch` alerts on is
ignored — `.gitignore:37` is `*.db` — so on 2026-10-02 the alert named the checkout's own
`workers.db` hourly for twelve hours while this instrument, the only thing on the machine
that can name a writer, reported `stray_count: 0` in 136 consecutive ledger rows. (The
alert prints that file by absolute path; this module may not repeat the spelling —
`tests/test_no_runtime_paths_in_code.py` refuses a home-qualified runtime path in
tracked code outside its own allowlist, and a docstring line is in that corpus while a
`#` comment is not.) A stray the tree's own ignore rules hide is still a stray
written into the code tree, so the read asks git the wider question. The one word that
keeps that affordable is `matching`: a wholly-ignored directory is reported as itself,
not enumerated, which is what stops the answer becoming the 367k-path `node_modules`
set. The live checkout measures 45 lines in 0.007 s, all of them `!!`, with `.venvs/`
standing as a single entry.

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
#:
#: ONE constant, and both readers use it: `scripts/automod/round.py` takes the
#: baseline, `scripts/automod/gate.py` takes the after-view. That is the whole reason
#: `--ignored=matching` goes here and nowhere else. A second command for one side only
#: would make the pair asymmetric, and the asymmetry is an accusation — the after-read
#: seeing `workers.db` while the baseline could not would credit every round with
#: every ignored store the tree already held.
#:
#: `=matching`, not a bare `--ignored`: bare `--ignored` with `-uall` descends into an
#: ignored directory and prints a line per file, which is the vendored-tree blow-up
#: `agent-services/guardian/datawatch.py`'s `_top_level` docstring records (367k paths).
#: `matching` prints the directory once.
UNTRACKED_CMD = ("git", "-C", "{root}", "status", "--porcelain",
                 "--untracked-files=all", "--ignored=matching")

#: A `git` failure is not an empty tree. `None` means "the tree could not be read",
#[] and every caller must treat that as unknown rather than as clean.
UNREADABLE = None


def untracked(root: Path | str) -> set[str] | None:
    """The paths `git status` calls untracked OR ignored under `root`, or None if git
    refused.

    `--untracked-files=all` matters: the default collapses a directory to one line,
    and a fixture written as `.t/05431072c3/r1873/` plus nine files inside it would
    diff as one path, which is one fewer path than anyone can then go and look at.
    Nine entries name nine files, and the one the round actually wrote is among them.

    `--ignored=matching` adds the other half of the stray class (#2059), and it is
    asymmetric in a way the caller should know: git lists an ignored FILE on its own
    line, so `workers.db` and `usage.db` are attributed file by file, while a directory
    whose contents are wholly ignored arrives as the directory — `!! .venvs/`, one
    entry. A stray written inside a wholly-ignored directory is therefore credited at
    directory granularity, not per file. That is the exact trade the flag buys: the
    class `datawatch` alerts on is top-level ignored files, and the class it would cost
    to enumerate is the vendored tree.

    A non-zero exit returns None rather than raising. This runs inside a round's
    promotion, and a tree that cannot be read must cost a piece of attribution, not
    the promotion — but it must say so, which is what None is for. The flag does not
    soften that: a `git` that fails on the wider question is just as unreadable as one
    that fails on the narrow one, and an empty set here would read as a clean tree.
    """
    argv = [part.format(root=str(root)) for part in UNTRACKED_CMD]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return UNREADABLE
    if proc.returncode != 0:
        return UNREADABLE
    return parse_porcelain(proc.stdout)


#: The porcelain statuses that mean "a path in the tree that git is not carrying":
#: untracked, and ignored (#2059 — the second is the class `datawatch` alerts on and
#: this parser used to discard). Both are strays; ` M`, `M `, `A `, `R ` are not.
STRAY_STATUSES = frozenset({"??", "!!"})


def parse_porcelain(text: str, statuses: frozenset[str] | None = None) -> set[str]:
    """The stray paths — untracked and ignored — out of `git status --porcelain` output.

    `statuses` narrows the answer to some of `STRAY_STATUSES` (`ignored` asks for
    `!!` alone); None is both, which is every caller before it.

    Kept separate from the subprocess so the parser is testable on recorded output —
    and so a rename or a quote-escaped path is a parsing question with a text answer,
    not something only reproducible by laying a working tree.

    The two-character status field is the contract, and the two statuses that answer
    this question are `STRAY_STATUSES`: `??` untracked and `!!` ignored. Everything
    else (` M`, `M `, `A `, `R `) is a tracked file and belongs to a different
    question — a modified file is not a stray, and crediting one to a round would
    report an ordinary edit as a write into the code tree. Splitting on the first space
    would lose a leading space entirely — `" M modified.py"` starts with one — which is
    how a modified file ends up looking untracked.
    """
    out: set[str] = set()
    for line in text.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:]
        if status in (STRAY_STATUSES if statuses is None else statuses):
            # A path git had to quote is printed `"with spaces"`; the quotes are
            # git's, not part of the name, and leaving them in produces a path
            # nobody can `ls`.
            out.add(path[1:-1] if path.startswith('"') and path.endswith('"') else path)
    return out


#: Path components that are a build or test cache wherever they appear. Ignored, and
#: created by ordinary work — running a module for the first time writes a
#: `__pycache__/` beside it — so they are not what `ignored` is asked about.
CACHE_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
                         "node_modules", ".venvs", ".vite", "graphify-out"})


def ignored(root: Path | str) -> set[str] | None:
    """The paths under `root` that git IGNORES, caches left out, or None if git refused.

    The per-call question the Bash tool asks (`agent_mcp/_bash_tree_strays.py`), and
    narrower than `untracked` on purpose. A `??` path is a file a job may be about to
    `git add` — a nightly task writing a new module is doing its job — while an ignored
    path can never be committed: in the live checkout it is runtime data or scratch
    that belongs in the data root. That is the class a wrong path creates
    (`sqlite3 workers.db` run from the tree, 2026-10-02) and the one nothing reports
    until the guardian's hourly check.

    `--no-optional-locks` because this runs beside the command it is measuring: a
    plain `git status` may take `index.lock` to refresh the index, and a worker's own
    `git commit` in the same second would fail on it.
    """
    argv = ["git", "--no-optional-locks",
            *(part.format(root=str(root)) for part in UNTRACKED_CMD[1:])]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return UNREADABLE
    if proc.returncode != 0:
        return UNREADABLE
    return {path for path in parse_porcelain(proc.stdout, frozenset({"!!"}))
            if not CACHE_PARTS.intersection(path.rstrip("/").split("/"))}


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
    """Record the whole stray set — untracked and ignored — uncapped, as this
    round's baseline.

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
    """The stray paths — untracked or ignored — that exist NOW and were not at the
    baseline.

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
