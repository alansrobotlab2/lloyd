"""/tmp headroom: alert before the tmpfs runs out of inodes, and say who holds them.

`/tmp` on this box is a tmpfs with a FIXED inode budget (1,048,576) that is
independent of its 126 GB size, so `df -h` reads 1% used while every
`mkdir` fails with ENOSPC. It ran out twice, and both times the production
tree was deleted:

* 2026-09-22 — attributed to a test-fixture teardown; `/tmp/pytest-of-<user>`
  held 944,339 files.
* 2026-09-29 — `mktemp -d` failed with an empty stdout, the gate's static rung
  read that as `Path("")` = its cwd = `~/lloyd`, and rmtree'd it
  (`architecture/testing.md` "2026-09-29"). At the time: pytest-of 666,565,
  Claude Code's own scratch 245,930, ~40 `/tmp/lloyd-check*`-style worktrees of
  ~2,300 each.

Nothing watched this. `df -i` is the one reading that shows it, and nobody runs
it until something has already failed. This module is evidence and an alarm,
never a deleter: it measures with `statvfs`, and only when over the threshold
walks the top level (bounded) to name the biggest consumers. Stdlib only, like
everything the guardian stages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

PATH = "/tmp"
#: Fraction of inodes OR blocks used at which the alert fires, escalates, and
#: clears. The clear is below the fire so a reading that hovers at the line
#: does not raise and retract an incident every tick.
WARN_FRACTION = 0.80
CRITICAL_FRACTION = 0.95
CLEAR_FRACTION = 0.70
#: The breakdown walk stops after this many entries: at a full budget a complete
#: `du --inodes` of /tmp is a million stats, which is not a guardian tick.
SCAN_LIMIT = 300_000
TOP_N = 8

ALERT_TITLE = "/tmp is running out of inodes"


@dataclass
class Reading:
    path: str
    inodes_total: int
    inodes_used: int
    bytes_total: int
    bytes_used: int

    @property
    def inode_fraction(self) -> float:
        return self.inodes_used / self.inodes_total if self.inodes_total else 0.0

    @property
    def byte_fraction(self) -> float:
        return self.bytes_used / self.bytes_total if self.bytes_total else 0.0

    @property
    def fraction(self) -> float:
        return max(self.inode_fraction, self.byte_fraction)


def measure(path: str = PATH, statvfs=os.statvfs) -> Reading:
    st = statvfs(path)
    return Reading(path=path,
                   inodes_total=int(st.f_files),
                   inodes_used=int(st.f_files - st.f_ffree),
                   bytes_total=int(st.f_blocks * st.f_frsize),
                   bytes_used=int((st.f_blocks - st.f_bfree) * st.f_frsize))


def level_for(fraction: float) -> str | None:
    if fraction >= CRITICAL_FRACTION:
        return "critical"
    if fraction >= WARN_FRACTION:
        return "error"
    return None


def top_consumers(path: str = PATH, limit: int = SCAN_LIMIT,
                  top_n: int = TOP_N) -> tuple[list[tuple[str, int]], bool]:
    """`([(name, entries), ...], truncated)` for the top-level entries of `path`.

    Counts directory entries below each top-level name without following
    symlinks or crossing devices, stopping after `limit` entries in total —
    the counts are then lower bounds and `truncated` says so."""
    counts: dict[str, int] = {}
    seen = 0
    try:
        dev = os.stat(path).st_dev
        tops = list(os.scandir(path))
    except OSError:
        return [], False
    truncated = False
    for top in tops:
        if seen >= limit:
            truncated = True
            break
        n = 1
        seen += 1
        try:
            if top.is_dir(follow_symlinks=False) and top.stat(follow_symlinks=False).st_dev == dev:
                stack = [top.path]
                while stack and seen < limit:
                    try:
                        with os.scandir(stack.pop()) as it:
                            for e in it:
                                n += 1
                                seen += 1
                                if e.is_dir(follow_symlinks=False):
                                    stack.append(e.path)
                                if seen >= limit:
                                    truncated = True
                                    break
                    except OSError:
                        continue
                if stack:
                    truncated = True
        except OSError:
            pass
        counts[top.name] = n
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return ranked, truncated


@dataclass
class TmpWatch:
    """Latched: fires on the first crossing and on an escalation, then stays
    quiet until the reading falls below `CLEAR_FRACTION`."""
    path: str = PATH
    statvfs: object = os.statvfs
    open_level: str | None = None
    last: Reading | None = field(default=None, repr=False)

    def tick(self) -> tuple[str, str | None, str]:
        """`(action, level, body)`; action is `alert`, `resolve` or `none`."""
        r = measure(self.path, self.statvfs)
        self.last = r
        level = level_for(r.fraction)
        if level and (self.open_level is None
                      or (level == "critical" and self.open_level != "critical")):
            self.open_level = level
            return "alert", level, self.describe(r)
        if self.open_level and r.fraction < CLEAR_FRACTION:
            self.open_level = None
            return "resolve", None, (f"{self.path} is back to {r.inodes_used:,} of "
                                     f"{r.inodes_total:,} inodes ({r.inode_fraction:.0%}).")
        return "none", level, ""

    def describe(self, r: Reading) -> str:
        top, truncated = top_consumers(self.path)
        lines = [
            f"{r.path}: {r.inodes_used:,} of {r.inodes_total:,} inodes used "
            f"({r.inode_fraction:.0%}); {r.bytes_used / 2**30:.1f} of "
            f"{r.bytes_total / 2**30:.0f} GiB ({r.byte_fraction:.0%}).",
            "At 100% every mkdir under /tmp fails with ENOSPC while `df -h` looks "
            "healthy — the state both tree deletions (09-22, 09-29) happened in.",
            "",
            "Largest top-level entries" + (" (lower bounds, scan capped)" if truncated else "") + ":",
        ]
        lines += [f"  {n:>9,}  {r.path}/{name}" for name, n in top] or ["  (could not list)"]
        lines += ["",
                  "Look before deleting: `df -i /tmp`, then "
                  "`du --inodes -x -d1 /tmp | sort -n | tail`. pytest's own retained "
                  "runs are `/tmp/pytest-of-$USER/pytest-N` (not `pytest-current`)."]
        return "\n".join(lines)
