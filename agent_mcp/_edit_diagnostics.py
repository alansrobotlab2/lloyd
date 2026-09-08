"""Post-edit Python diagnostics, appended to a successful Edit/Write result.

Why here rather than at the gate
--------------------------------
The selfmod gate already runs pyflakes as a delta, and it is the wrong place
to *learn* about a broken edit: it runs minutes later, after the model has
built ten more edits on top of the mistake. opencode's one mechanical
advantage over this harness was that its edit results carry diagnostics, so
the feedback loop closes at the edit. This is that, using the gate's own
normalisers so the two cannot disagree.

Three rules the delta has to follow, all of them learned from the gate:

* **Delta, never absolute.** The tree carries ~69 tolerated pyflakes
  findings. Reporting them on every edit would be pure noise, and worse,
  would train the model to skip the block.
* **Position is not identity.** `lint_findings.normalize_pyflakes_line`
  drops line and column, so inserting ten lines at the top of a file does
  not report every finding below as new.
* **A multiset, not a set.** Two normalised-identical findings are two
  findings; `Counter(post) - Counter(pre)` keeps the second one visible.

Nothing here may raise into the edit. Every entry point returns "" on any
failure, and the block is appended only to a *success* string — `text_result`
sniffs a leading JSON object with an "error" key to set `isError`, so
appending to an error payload would break the JSON and flip the flag.
"""

from __future__ import annotations

import logging
from collections import Counter

from app.lint_findings import normalize_pyflakes_line

logger = logging.getLogger("lloyd-edit-diagnostics")

# Either image above this is skipped. A generated or vendored file that size
# is not what the model just hand-edited, and pyflakes on it is seconds.
MAX_SOURCE_BYTES = 1_000_000

DEFAULT_MAX_LINES = 30


class _Collector:
    """pyflakes Reporter that keeps findings instead of printing them.

    `pyflakes.api.check` calls `syntaxError` and returns *without* running
    the checker when the source does not parse, so `syntax` and `flakes` are
    never both populated — that asymmetry is the whole reason a syntax error
    gets its own block below.
    """

    def __init__(self) -> None:
        self.flakes: list[tuple[int, int, str]] = []
        self.syntax: tuple[int, int, str, str] | None = None   # line, col, msg, text
        self.unexpected: str = ""

    def unexpectedError(self, filename: str, msg: str) -> None:  # noqa: N802
        self.unexpected = str(msg)

    def syntaxError(self, filename: str, msg: str, lineno, offset, text) -> None:  # noqa: N802
        line = text.splitlines()[-1] if text else ""
        self.syntax = (int(lineno or 1), int(offset or 1), str(msg), line.rstrip())

    def flake(self, message) -> None:
        try:
            text = message.message % message.message_args
        except Exception:                       # pragma: no cover - defensive
            text = str(message)
        # pyflakes' col is 0-based; every other position in this file is the
        # 1-based number the model sees in a Read.
        self.flakes.append((int(message.lineno), int(message.col) + 1, text))


def _check(source: str, filename: str) -> _Collector:
    import pyflakes.api

    c = _Collector()
    pyflakes.api.check(source, filename, c)
    return c


def _normalised(collector: _Collector, filename: str) -> Counter:
    out: Counter = Counter()
    for lineno, col, text in collector.flakes:
        norm = normalize_pyflakes_line(f"{filename}:{lineno}:{col}: {text}")
        if norm:
            out[norm] += 1
    return out


def is_python(path: str) -> bool:
    """`.py` only. A `.pyi` stub is full of deliberately unused names."""
    return path.endswith(".py")


def python_block(path: str, pre_bytes: bytes | None, post_text: str,
                 max_lines: int = DEFAULT_MAX_LINES) -> str:
    """The `<diagnostics>` block for one edit, or "" when there is nothing new.

    `pre_bytes` is None for a created file, in which case everything the new
    file reports is new — there is no baseline to subtract.
    """
    try:
        return _python_block(path, pre_bytes, post_text, max_lines)
    except Exception:
        logger.warning("edit diagnostics failed for %s", path, exc_info=True)
        return ""


def _python_block(path: str, pre_bytes: bytes | None, post_text: str,
                  max_lines: int) -> str:
    if not is_python(path):
        return ""
    if len(post_text.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return ""
    if pre_bytes is not None and len(pre_bytes) > MAX_SOURCE_BYTES:
        return ""

    post = _check(post_text, path)

    if post.syntax is not None:
        pre_broken = False
        if pre_bytes is not None:
            pre = _check(pre_bytes.decode("utf-8", errors="replace"), path)
            pre_broken = pre.syntax is not None
        return _syntax_block(path, post.syntax, pre_broken)

    if pre_bytes is None:
        # A created file has no baseline: every finding is new.
        pre_counts: Counter = Counter()
    else:
        pre = _check(pre_bytes.decode("utf-8", errors="replace"), path)
        if pre.syntax is not None:
            # The pre-image did not parse, so pyflakes never ran on it and
            # there is no flake baseline to subtract. Reporting the whole
            # post-image here would dump every tolerated finding in the file
            # onto an edit that just *fixed* the syntax.
            return ""
        pre_counts = _normalised(pre, path)

    post_counts = _normalised(post, path)
    new = post_counts - pre_counts
    if not new:
        return ""

    # Display uses post-image positions, which is what the model can act on.
    remaining = Counter(new)
    lines: list[str] = []
    for lineno, col, text in sorted(post.flakes):
        norm = normalize_pyflakes_line(f"{path}:{lineno}:{col}: {text}")
        if norm and remaining.get(norm, 0) > 0:
            remaining[norm] -= 1
            lines.append(f"{lineno}:{col}: {text}")

    total = len(lines)
    if max_lines > 0 and total > max_lines:
        lines = lines[:max_lines] + [f"... and {total - max_lines} more"]
    body = "\n".join(lines)
    return (f'<diagnostics file="{path}" tool="pyflakes" new="{total}">\n'
            f'{body}\n</diagnostics>')


def _syntax_block(path: str, syntax: tuple[int, int, str, str],
                  pre_broken: bool) -> str:
    lineno, col, msg, text = syntax
    tag = ' pre_existing="true"' if pre_broken else ""
    body = f"{lineno}:{col}: {msg}"
    if text:
        body += f"\n{text}"
    return (f'<diagnostics file="{path}" tool="pyflakes" syntax_error="true"'
            f'{tag}>\n{body}\n</diagnostics>')


def config() -> dict:
    """`harness.edit_diagnostics`, read at call time so tests can patch it."""
    defaults = {"python": True, "typescript": True, "max_lines": DEFAULT_MAX_LINES}
    try:
        from app.config import CONFIG
        raw = (CONFIG.get("harness") or {}).get("edit_diagnostics") or {}
    except Exception:
        raw = {}
    if isinstance(raw, dict):
        defaults.update({k: v for k, v in raw.items() if v is not None})
    return defaults
