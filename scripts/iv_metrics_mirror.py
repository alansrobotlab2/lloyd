#!/usr/bin/env python3
"""Mirror the Inner Voice metrics series into the vault — and never shrink it (backlog #2121).

The nightly job (#86) has copied
`~/lloyd-data/_pipeline/reflection/iv-metrics.jsonl` over
`~/obsidian/knowledge/inner-voice/iv-metrics.jsonl` with a bare `cp` since the copy step
existed. The vault is the only place that history is committed: `_pipeline/` is gitignored
(`.gitignore:34`), so the copy is what makes a nightly number checkable after the fact
(#1229). A whole-file copy of a file that is supposed to be append-only is only safe while
the source really is the longer file.

On 2026-09-23 it was not. The data-root cutover moved the series to a new path, so the `cp`
read a 1-row file at the new location and wrote it over the 11-row mirror: 10 rows of the
observer's measurement history left the only committed copy of that history. Nothing
refused, and nothing logged a refusal — the rows were restored from HEAD as vault commit
`e508d468` before anything was committed, which is why `git log --numstat` on that path
still reads `+N/-0` for every commit including the recovery. A future shrink has no reason
to be noticed by anything except this script.

Why a count, and why non-zero
----------------------------
`wc -l source < wc -l mirror` is the check the incident fails (1 < 11), and it is the check
this script runs. The refusal **exits 1**: the caller is an autonomy run that reads its own
exit code out of a Bash tool result. `scripts/backup/backup-vault.sh:64` refuses a shrunken
vault with `exit 0`, which is right for a systemd unit whose only reader is the journal and
wrong here — a nightly that "succeeded" while its guard refused is the same silent green
night #460 was filed to end. One refusal code for every refusal, because the printed token
is what the run is told to put verbatim in its report (`skills/iv-metrics-series/SKILL.md`
step 3), and a second code would put the text `exit code 2` into the same report where step
4 reserves that token for the recorder's breach.

The refusal tokens are what a report carries, not the counts:

  `mirror-shrunk`   the source holds fewer rows than the mirror already does
  `source-missing`  the source does not exist — e.g. a moved path, the 09-23 root cause
  `source-empty`    the source exists and holds zero rows — a night the grader emitted
                    nothing, which the recorder itself exits 3 on
  `copy-failed`     the guard passed and the write did not (missing permissions, full disk)

What this deliberately does NOT check
-------------------------------------
An append-only prefix test (the mirror's rows are a prefix of the source's) would catch one
case this count misses — an equal-length source from a renamed path. It is not here,
because it fails the other way and harder: any legitimate rewrite of the source — a
restored file, a compaction, a row order a later tool changed — makes the source not a
prefix, and a prefix guard then refuses EVERY later night. A guard that refuses nightly is
indistinguishable from a mirror that is current, and the failure it produces is a mirror
that silently stops updating forever. Acceptance clause 2 also states the property as
"line count at or above the mirror's → copy it whole", which a prefix test contradicts. The
count is the guard; the vault's git history is the recovery.

Rows, not newlines
------------------
`row_count` counts rows, so a newest row that has not had its trailing newline written yet
still counts as one. `wc -l` counts newlines and would call that file a row shorter than it
is, which in the source's direction is a phantom shrink: the copy would be refused, then
refused again the next night, and the mirror would stop updating with a green run beside it
every time.

Exit codes: 0 copied (or nothing changed) · 1 refused, for any of the four reasons above.
Writes nothing except the mirror, and never to `usage.db` — same contract as the recorder.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The one owner of each layout fact: the recorder's own `--out` default is
# `PIPELINE_DIR / "reflection" / "iv-metrics.jsonl"` (`scripts/iv_metrics_record.py:635`),
# and `vault_root()` honours `LLOYD_VAULT_ROOT` so a round can point the mirror at a copy
# instead of the live vault (`app/data_root.py:199-211`).
from app.data_root import vault_root  # noqa: E402
from app.paths import PIPELINE_DIR  # noqa: E402

SERIES_REL = ("reflection", "iv-metrics.jsonl")
DEFAULT_SOURCE = PIPELINE_DIR.joinpath(*SERIES_REL)
DEFAULT_MIRROR = vault_root() / "knowledge" / "inner-voice" / "iv-metrics.jsonl"

EXIT_OK = 0
EXIT_REFUSED = 1

SHRUNK = "mirror-shrunk"
MISSING = "source-missing"
EMPTY = "source-empty"
COPY_FAILED = "copy-failed"


def row_count(path: Path) -> int | None:
    """Rows in a JSONL file, or None if there is no such file.

    Iterating a binary handle splits on `\\n` AND yields a trailing partial chunk, so this
    is `wc -l`'s count plus a newest row that has not been newline-terminated yet. See the
    module docstring for why that difference is the difference between a guard and a lock.
    """
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _refuse(token: str, detail: str, mirror: Path) -> int:
    """One refusal line on stderr, naming the token and the untouched mirror."""
    print(f"iv-metrics-mirror: {token} — {detail}. Nothing was written; the mirror at "
          f"{mirror} is unchanged. Quote {token} verbatim in your report.",
          file=sys.stderr)
    return EXIT_REFUSED


def mirror(source: Path, mirror_path: Path) -> int:
    """Copy `source` over `mirror_path` whole, unless that would destroy history."""
    source_rows = row_count(source)
    if source_rows is None:
        return _refuse(MISSING, f"source {source} does not exist (a moved data root reads "
                                f"exactly like a quieter night)", mirror_path)
    if source_rows == 0:
        return _refuse(EMPTY, f"source {source} holds 0 rows — the recorder exits 3 on "
                              f"exactly this night and appends nothing", mirror_path)

    mirror_rows = row_count(mirror_path) or 0
    if source_rows < mirror_rows:
        return _refuse(
            SHRUNK,
            f"refusing to copy source={source_rows} mirror={mirror_rows} rows from {source}: "
            f"the mirror holds {mirror_rows - source_rows} row(s) the source does not, which "
            f"is the 2026-09-23 failure (1 over 11, 10 rows lost, restored as e508d468); "
            f"recover them from the vault's git history, do not overwrite them",
            mirror_path)

    # Write-then-replace, so a copy interrupted half way cannot leave a truncated mirror
    # standing — the exact state this script exists to refuse.
    part = mirror_path.with_name(mirror_path.name + ".part")
    try:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(source.read_bytes())
        os.replace(part, mirror_path)
    except OSError as exc:
        part.unlink(missing_ok=True)
        return _refuse(COPY_FAILED, f"{type(exc).__name__}: {exc}", mirror_path)

    print(f"iv-metrics-mirror: copied source={source_rows} mirror_was={mirror_rows} rows "
          f"({source} -> {mirror_path})")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="iv_metrics_mirror.py",
        description="Guarded whole-file mirror of the IV metrics series into the vault "
                    "(#2121). Refuses rather than shrinking the mirror.")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE),
                    help="the live series (default: the recorder's own --out path)")
    ap.add_argument("--mirror", default=str(DEFAULT_MIRROR),
                    help="the vault copy to overwrite (default: knowledge/inner-voice/)")
    args = ap.parse_args(argv)
    return mirror(Path(args.source).expanduser(), Path(args.mirror).expanduser())


if __name__ == "__main__":
    raise SystemExit(main())
