"""Which qmd hit stands for a backlog item — and which is only a file in the
same collection.

`~/.config/qmd/index.yml` indexes the board as
`backlog: {path: /home/alansrobotlab/obsidian/backlog, pattern: "**/*.md"}`,
and `**` is recursive. The real tasks are flat `{id}-{slug}.md` at the top
level; `backlog/data/` holds run witnesses and dry-run records, and on
2026-10-13 all 39 of its `.md` files were in the live index with chunk-0
vectors, which is exactly the row set `scripts/automod/cluster.py`'s
`load_vectors` selects.

Both consumers that mapped a hit's path back to an id took the digits from any
path component, so every date-stamped witness was task **#2026**:
`agent_mcp/backlog_similar.py`'s `_QMD_ID_RE` had a `(?:^|/)` start anchor, and
`scripts/automod/cluster.py` matched `os.path.basename(path)`, which discards
the directory that says a row is not a task. Measured on the running daemon on
2026-10-13: a `vec`+`rerank` query over the text of
`backlog/data/2026-10-02.2047-mirror-refresh-deadlock-witness.md` answers that
witness at reranker **1.000** and `semantic_candidates` returned
`{'id': 2026, 'score': 1.0}` — a score no item's own text earned, spending one
of the six slots that feed the merge decision and handing `closed_match` a
rule-A hit against closed #2026, which is the refusal path (#1051).

So the flat layout is a **read-side** contract, not just the write-side
convention `architecture/backlog.md`'s Storage section documents: an id is
taken only from a path that is a direct child of the collection root. Anything
in another spelling — a subdirectory, another collection, an absolute
filesystem path — yields no id, which costs an advisory neighbour and never a
wrong merge target (#2657).
"""

from __future__ import annotations

import re

QMD_SCHEME = "qmd://"
BACKLOG_COLLECTION = "backlog"

# The board's own id rule, applied to a name already known to be at the root.
_FLAT_ID_RE = re.compile(r"^(\d+)[-_].*\.md$")


def backlog_hit_id(path: object) -> int | None:
    """The item id a `backlog`-collection hit stands for, or None.

    Accepts the two forms the board's two readers meet, both measured on the
    live index on 2026-10-13: the daemon's reply, `qmd://backlog/2054-x.md`
    (URI-decoded already, by `agent_mcp.vault.qmd_query`), and qmd's own stored
    row, relative to the collection root with no directory at all —
    `SELECT path FROM documents WHERE collection='backlog' AND path NOT LIKE
    '%/%'` returns `10-model-routing-based-on-keyword-for-now.md`, not
    `backlog/10-…`. A hit whose remaining path still has a `/` is a file in a
    subdirectory — a `backlog/data/` witness — and an id read from it belongs
    to the year in its filename, not to a task.
    """
    s = str(path or "").strip()
    if not s:
        return None
    if s.startswith(QMD_SCHEME):
        # Everything after the scheme is `<collection>/<path relative to it>`,
        # so the first component is the collection's NAME, not a directory
        # inside the board. Another collection's hit is not a task here.
        collection, sep, relative = s[len(QMD_SCHEME):].partition("/")
        if collection != BACKLOG_COLLECTION:
            return None
        s = relative if sep else collection
    if "/" in s:
        return None
    m = _FLAT_ID_RE.match(s)
    return int(m.group(1)) if m else None
