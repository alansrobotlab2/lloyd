"""The pyflakes rail for `.py` files a foreground Bash command rewrote (#695).

`Edit` and `Write` carry a `<diagnostics>` block for what they introduced;
until this, a file mutated through Bash — `sed -i`, `patch`/`git apply`, a
`> x.py` redirect, a heredoc doing `Path("x.py").write_text(...)`, `black`,
`ruff --fix` — carried nothing, and the implementer learned of the break at
the automod gate. Over two weeks to 2026-09-15 223 Bash commands wrote a
repo `.py`, the majority from automod rounds, while 279 more Bash calls
re-derived the missing delta by hand.

Two halves, both off the event loop and both unable to fail the command:

- `snapshot(command, cwd)` runs *before* the shell starts. Only a command
  that carries a write marker is looked at at all, so `grep`/`cat` cost
  nothing. Its `.py` tokens are resolved against the cwd and every `cd` in
  the command, kept when they sit inside a git work tree and outside
  `_pipeline/` and `sessions/`, and their current bytes are the pre-image.
- `append(text, snap)` re-reads each one afterwards and hands the files whose
  bytes changed to `_edit_diagnostics.python_block` — the same delta `Edit`
  gets, so a tolerated finding already in the file is never reported. An
  absolute block (no pre-image) fired on 120 of 531 tracked files when
  measured, which is the noise that trains a model to skip the block.

Extraction only bounds which files are read; the byte comparison decides
what is reported. A token that names a file the command merely read (a
`read_text()` heredoc, the input of a `grep … > /tmp/out`) is unchanged
afterwards and produces nothing.

Not covered, deliberately: background Bash (its result is a task id and
cannot carry a block), sandboxed sessions (read-only by construction), the
code-graph and tsc rails, and the change ledger, which still sees Edit/Write
only.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger("lloyd-builtin-bash")

MAX_TARGETS = 32          # candidate files read per command
MAX_PATCH_BYTES = 1_000_000

# Anything that can write a file. `>` excludes fd duplication (`2>&1`, `>&2`).
_WRITE_MARKER = re.compile(
    r"""
      \bsed\b[^|;&\n]*\s(?:-[a-zA-Z]*i|--in-place)
    | \bperl\b[^|;&\n]*\s-[a-zA-Z]*i
    | (?<![0-9&>])>>?(?!&)
    | \btee\b
    | \bpatch\b
    | \bgit\s+apply\b
    | \b(?:black|ruff|isort|autoflake|autopep8|yapf)\b
    | \.write_(?:text|bytes)\s*\(
    | \bopen\s*\([^)]*,\s*(?:mode\s*=\s*)?['"][rwxab+]*[wxa]
    | \b(?:cp|install)\s
    """,
    re.X,
)
_PY_TOKEN = re.compile(r"""([^\s'"`;|&<>(){}\[\],=:]+\.py)(?![\w.])""")
_PATCH_FILE = re.compile(r"""([^\s'"`;|&<>(){}\[\],=:]+\.(?:diff|patch))(?![\w.])""")
_CD = re.compile(r"""(?:^|[;&|\n(]\s*)cd\s+(['"]?)([^\s'";&|)]+)\1""")
_EXCLUDED_PARTS = frozenset({"_pipeline", "sessions"})


@dataclass
class Snapshot:
    # abs path -> pre-image bytes, or None for a file that did not exist yet
    pre: dict[str, bytes | None] = field(default_factory=dict)


def _expand(token: str) -> str:
    return os.path.expanduser(os.path.expandvars(token))


def _in_tree(path: str) -> bool:
    """Inside a git work tree, and not under a runtime-data directory."""
    if _EXCLUDED_PARTS.intersection(path.split(os.sep)):
        return False
    d = os.path.dirname(path)
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return True
        parent = os.path.dirname(d)
        if parent == d:
            return False
        d = parent


def _bases(command: str, cwd: str | None) -> list[str]:
    base = cwd or os.getcwd()
    out = [base]
    for _q, raw in _CD.findall(command):
        d = _expand(raw)
        d = d if os.path.isabs(d) else os.path.join(base, d)
        if os.path.isdir(d) and d not in out:
            out.append(os.path.normpath(d))
    return out


def _tokens(command: str, bases: list[str]) -> list[str]:
    found = list(_PY_TOKEN.findall(command))
    # A patch applied from a file names its targets inside the file.
    if re.search(r"\bpatch\b|\bgit\s+apply\b", command):
        for raw in _PATCH_FILE.findall(command):
            for base in bases:
                p = os.path.join(base, _expand(raw))
                try:
                    if os.path.getsize(p) <= MAX_PATCH_BYTES:
                        with open(p, encoding="utf-8", errors="replace") as fh:
                            found.extend(_PY_TOKEN.findall(fh.read()))
                        break
                except OSError:
                    continue
    return found


def _resolve(token: str, bases: list[str]) -> list[str]:
    tok = _expand(token)
    if os.path.isabs(tok):
        return [os.path.normpath(tok)]
    variants = [tok]
    # `+++ b/app/x.py` in a unified diff.
    if tok.startswith(("a/", "b/")):
        variants.append(tok[2:])
    hits = [os.path.normpath(os.path.join(b, v)) for b in bases for v in variants]
    existing = [h for h in hits if os.path.isfile(h)]
    # A file that does not exist yet may be what the command creates; the
    # last `cd` is where a relative redirect would land it.
    return existing or [os.path.normpath(os.path.join(bases[-1], tok))]


def snapshot(command: str, cwd: str | None) -> Snapshot | None:
    """Pre-images of the tree `.py` files this command may write, or None."""
    if not _WRITE_MARKER.search(command):
        return None
    bases = _bases(command, cwd)
    snap = Snapshot()
    for token in _tokens(command, bases):
        for path in _resolve(token, bases):
            if path in snap.pre or not path.endswith(".py") or not _in_tree(path):
                continue
            if len(snap.pre) >= MAX_TARGETS:
                break
            try:
                with open(path, "rb") as fh:
                    snap.pre[path] = fh.read()
            except FileNotFoundError:
                snap.pre[path] = None
            except OSError:
                continue
    return snap if snap.pre else None


def append(text: str, snap: Snapshot | None) -> str:
    """`text` plus a `<diagnostics>` block per changed file; never raises."""
    if snap is None:
        return text
    try:
        from agent_mcp import _edit_diagnostics as diag
        cfg = diag.config()
        if not cfg.get("python", True):
            return text
        max_lines = int(cfg.get("max_lines", diag.DEFAULT_MAX_LINES))
        blocks: list[str] = []
        for path, pre in snap.pre.items():
            try:
                with open(path, "rb") as fh:
                    post = fh.read()
            except OSError:
                continue
            if post == pre:
                continue
            block = diag.python_block(
                path, pre, post.decode("utf-8", errors="replace"), max_lines)
            if block:
                blocks.append(block)
        if blocks:
            return text + "\n\n" + "\n\n".join(blocks)
    except Exception:
        logger.warning("bash edit diagnostics failed", exc_info=True)
    return text

