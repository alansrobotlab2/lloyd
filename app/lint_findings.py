"""Normalisers shared by the automod gate and the aggregator's edit diagnostics.

Both answer the same question — "is this finding *new*?" — and they must
answer it the same way. The gate judges pyflakes and tsc as deltas against
the pre-image because the tree carries ~69 tolerated pyflakes findings and a
handful of tsc errors; an absolute bar would fail every round forever. The
post-edit diagnostics appended to an Edit result have to use those same
normalisers or the model is told about a finding the gate will not mind, or
worse, not told about one it will.

This lives in `app/` and imports nothing, because the aggregator cannot
import `scripts.automod.gate` — that module pulls in the whole
self-modification package (worktrees, promotion, ledger state), and
`agent_mcp` must not depend on any of it. `gate._pyflakes` and
`gate._parse_tsc` are thin wrappers over these functions, pinned
behaviour-identical by test.

Both normalisations drop the line and column deliberately: a finding that
merely moved down the file is not a new finding, and an edit that inserts
ten lines at the top would otherwise report every finding below it as new.
"""

from __future__ import annotations

import os
import re
from collections import Counter

# "path:LINE:COL: message" → "path: message"
PYFLAKES_LINE_RE = re.compile(r"^(.*?):\d+:\d+:\s*(.*)$")

# "path(LINE,COL): error TSnnnn: message" → "path: error TSnnnn: message"
TSC_LINE_RE = re.compile(r"^(.*?)\(\d+,\d+\):\s*(error TS\d+:.*)$")


def normalize_pyflakes_line(line: str) -> str | None:
    """One pyflakes output line as a comparable finding, or None if blank.

    A line that does not match the `path:line:col: message` shape (pyflakes
    prints bare text for some internal failures) is kept verbatim rather
    than dropped — an unparseable finding is still a finding.
    """
    s = line.strip()
    if not s:
        return None
    m = PYFLAKES_LINE_RE.match(s)
    if m:
        return f"{m.group(1)}: {m.group(2)}"
    return s


def parse_pyflakes(text: str) -> set[str]:
    """Every finding in a pyflakes run, normalised.

    A set, not a multiset: pyflakes reports one finding per line per name, so
    two identical normalised lines in one file are the same problem seen at
    two line numbers, and the gate has always compared these as a set.
    """
    out: set[str] = set()
    for line in text.splitlines():
        norm = normalize_pyflakes_line(line)
        if norm:
            out.add(norm)
    return out


def parse_tsc(text: str) -> Counter:
    """Every tsc error, normalised and counted.

    A multiset rather than a set: a second copy of an existing error in the
    same file is a new error, and a set would hide it.
    """
    out: Counter = Counter()
    for line in text.splitlines():
        m = TSC_LINE_RE.match(line.strip())
        if m:
            out[f"{m.group(1)}: {m.group(2)}"] += 1
    return out


def split_tsc_by_file(counter: Counter) -> dict[str, Counter]:
    """Group normalised tsc findings by the path they name.

    The whole-project run is the only one tsc offers (tsconfig `include`
    covers `src`), so a per-file delta has to be carved out of a whole-project
    result. The path is everything before the first ": " — the same split the
    normaliser produced.
    """
    by_file: dict[str, Counter] = {}
    for finding, n in counter.items():
        path = finding.split(": ", 1)[0]
        by_file.setdefault(path, Counter())[finding] += n
    return by_file


def node_env() -> dict:
    """Environment for running a node binary out of `web/node_modules/.bin`.

    The PATH prefix is needed because the supervisord environment does not
    carry the system bin dirs; NODE_OPTIONS is dropped because whatever the
    parent set it to (a debugger, a loader) is not what a type-check wants.
    """
    env = dict(os.environ)
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")
    env.pop("NODE_OPTIONS", None)
    return env
