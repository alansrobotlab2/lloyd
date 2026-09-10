"""Post-edit Python diagnostics, appended to a successful Edit/Write result.

Why here rather than at the gate
--------------------------------
The automod gate already runs pyflakes as a delta, and it is the wrong place
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

There is a second block here, and it exists for the same reason. pyflakes is
per-file by construction, so the block above is blind to the failure class
that motivated it: a signature changes, the edited file still parses and
lints clean, and a caller in another module is never looked at. Lloyd has
cross-file knowledge — `agent_mcp/code_graph.py`, the store the `graph_*`
tools read — but only as a *pull*, which fires when the model remembers to
ask. `callers_block()` makes it passive: an edit that changes a module-level
interface carries the inbound callers that live outside the file. See
"Cross-file blast radius" below for the rules that keep it quiet.

Nothing here may raise into the edit. Every entry point returns "" on any
failure, and the block is appended only to a *success* string — `text_result`
sniffs a leading JSON object with an "error" key to set `isError`, so
appending to an error payload would break the JSON and flip the flag.

One constraint the second block adds: `_append_diagnostics` runs on the event
loop, not in the edit's worker thread, so a graph read here would stall SSE
chat and voice as well. `code_graph` loads 12 MB of JSON in ~0.2 s, which is
over budget, so the whole graph half runs in a daemon thread the edit joins
for at most `RAIL_BUDGET_S` — too slow means no block, never a slow edit. The
load lands in a per-root cache the next edit reuses, and the lookup after a
load is microseconds.
"""

from __future__ import annotations

import ast
import logging
import os
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

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


# ── cross-file blast radius ────────────────────────────────────────────────
#
# The rules that make the pyflakes delta worth reading apply here too, and
# the cheapest one is the same: a block that fires on every edit is a block
# the model learns to skip, which is worse than not having one.
#
# * **Only an interface change fires.** A module-level name, signature,
#   base, assigned value, or a `return`/`yield` expression. A local variable
#   in a body changes nothing outside the file, so it must add nothing —
#   that is what keeps the rail off ~most edits.
# * **Only the pre-image's symbols are queried.** The graph is a picture of
#   the tree *before* the edit, which is exactly the right source for "who
#   calls this today" — a stale graph is correct here, not a hazard. A name
#   that only exists after the edit cannot resolve, and that silence is the
#   design: querying a brand-new name would match some unrelated symbol of
#   the same spelling in another file and report a caller that never existed.
# * **A rename is caught from the old side.** `fetch_user` → `fetch_user_v2`
#   means the pre-image's `fetch_user` is gone from the post-image, which is
#   a touched symbol, and the callers still spell it the old way.
# * **Textual reach is not type checking.** A bool→dataclass return change
#   produces no finding in pyflakes or the graph; what this can say is "these
#   call sites touch the thing you just changed", and judging each one stays
#   with the model. Hence advisory wording, never an error.
# * **Hard caps, and a fan-out ceiling.** A widely imported helper has
#   callers everywhere; listing 40 of them is a wall of text that gets
#   skipped, so a symbol above `FANOUT_CEILING` call sites is dropped and the
#   drop is stated in one line.

# Measured on this tree, so the two numbers below are a sum and not a guess.
# The rail pays for two AST passes before it touches the graph, and that pass
# costs ~0.28 ms/KB: 15 ms at 44 KB (code_graph.py), 27 ms at 110 KB, 46 ms at
# 124 KB, 64 ms at 232 KB. A warm lookup afterwards is under 1 ms. The graph
# load is 210 ms, and `import agent_mcp.code_graph` is 516 ms in a cold
# process but a sys.modules hit inside the aggregator, which imported
# `mcp.types` long ago.
#
# So: 90 ms of join + ~65 ms of AST at the cap keeps the worst case near
# 155 ms of *rail*, of which the edit only ever waits the join — and the AST
# pass is work the pyflakes half was already doing on the same images. A
# 120 ms join measured 153-169 ms end to end, which is over the promise.
# Files above RAIL_MAX_SOURCE_BYTES get no blast-radius block: at 400 KB the
# AST pass alone is ~220 ms, and pyflakes still runs on them.
RAIL_BUDGET_S = 0.09          # what one edit may pay; the promise is < 150 ms
RAIL_MAX_SOURCE_BYTES = 120_000   # ~65 ms of AST; MAX_SOURCE_BYTES is 3× that
FANOUT_CEILING = 20           # call sites for one symbol before it is dropped
MAX_CALLER_FILES = 5
MAX_CALLER_LINES = 10
MAX_SYMBOLS = 8               # interfaces one edit may change at once
MAX_CACHED_GRAPHS = 2         # an Entry holds a whole networkx graph
_MISS_TTL_S = 300.0           # don't re-read an unreadable graph every edit

_GRAPHS: "OrderedDict[str, tuple[int, int, Any]]" = OrderedDict()
_LOADING: dict[str, threading.Event] = {}
_MISS: dict[str, float] = {}
_GRAPH_LOCK = threading.Lock()


def _forget_all_graphs() -> None:
    """Drop every cached graph. A test hook; this cache is process-wide."""
    with _GRAPH_LOCK:
        _GRAPHS.clear()
        _LOADING.clear()
        _MISS.clear()


# ── which symbols did this edit actually change? ────────────────────────────


def _dump(node: Any) -> str:
    try:
        return ast.dump(node)
    except Exception:                                     # pragma: no cover
        return f"<undumpable {type(node).__name__}>"


def _returns(body: list) -> list[str]:
    """Dumps of every value this symbol itself can hand back.

    Nested defs, classes and lambdas are units of their own: a change inside
    a closure is not a change to this signature, and counting it would make
    every body edit look like an interface edit — which is the whole noise
    problem in one rule.
    """
    out: list[str] = []
    stack = list(body)
    while stack:
        node = stack.pop()
        # Test the node itself, not only its children: `return x` as a direct
        # statement of a body is a top-level statement, and a walker that
        # looks only at children misses every return in the common function.
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
            out.append(_dump(node))
            continue                      # a return holds no further returns worth naming
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return sorted(out)


def _def_fp(node: Any) -> str:
    """Name, signature, decorators, declared return, and actual returns."""
    parts = [type(node).__name__, node.name, _dump(node.args)]
    parts += [_dump(d) for d in node.decorator_list]
    if getattr(node, "returns", None) is not None:
        parts.append(_dump(node.returns))
    if getattr(node, "type_params", None):
        parts.append(_dump(node.type_params))
    return "|".join(parts) + "|" + "|".join(_returns(node.body))


def _class_fp(node: Any) -> str:
    """Class shape only: decorators, bases, and the non-method body.

    Methods are `Class.method` symbols in their own right, so leaving them
    out is what stops one method edit from reporting every importer of the
    class as if the class had changed.
    """
    parts = ["class", node.name]
    parts += [_dump(d) for d in node.decorator_list]
    parts += [_dump(b) for b in node.bases]
    parts += [_dump(k) for k in node.keywords]
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        value = getattr(stmt, "value", None)
        if (isinstance(stmt, ast.Expr) and isinstance(value, ast.Constant)
                and isinstance(value.value, str)):
            continue                                      # docstring
        parts.append(_dump(stmt))
    return "|".join(parts)


def _interface_symbols(source: str) -> dict[str, str] | None:
    """`{symbol: fingerprint}` for a module's public shape. None if unparseable.

    Keys are module-level `def`/`class`/assignment names and one level of
    `Class.method`. Never raises into the edit: `ast.parse` on a post-image
    that does not parse is the normal "say nothing" path.
    """
    try:
        tree = ast.parse(source)
    except Exception:
        return None
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = _def_fp(node)
        elif isinstance(node, ast.ClassDef):
            out[node.name] = _class_fp(node)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = _def_fp(sub)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = _dump(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = _dump(node)
    return out


def _touched_symbols(pre_text: str, post_text: str) -> list[str]:
    """Module-level names whose interface this edit changed, sorted.

    Restricted to names the *pre*-image carried, for the reason in the block
    comment above: they are the only names the graph can answer about.
    """
    post = _interface_symbols(post_text)
    if post is None:
        return []
    pre = _interface_symbols(pre_text) or {}
    return [n for n in sorted(pre) if pre[n] != post.get(n)]


# ── the graph, on a budget ──────────────────────────────────────────────────


def _code_graph() -> Any:
    """The code-graph module, or None. Imported lazily and off-thread:
    `code_graph` pulls in `mcp.types`, which costs ~0.5 s cold and has no
    business being paid on the event loop."""
    try:
        from agent_mcp import code_graph
        return code_graph
    except Exception:                                     # pragma: no cover
        logger.warning("code_graph unusable for edit diagnostics", exc_info=True)
        return None


def _line_of(loc: Any) -> int:
    try:
        return int(str(loc).lstrip("Ll") or 0)
    except (TypeError, ValueError):
        return 0


def _same_file(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a == b or a.endswith("/" + b)
                                    or b.endswith("/" + a))


def _rel_to(root: str, edited: str) -> str:
    """The edited file as the graph names it, or "" when it is outside root."""
    try:
        rel = os.path.relpath(edited, root)
    except (TypeError, ValueError):
        return ""
    return "" if rel.startswith((".", "..")) else rel


def _root_for(cg: Any, edited: str, fallback_root: str) -> str:
    """The tree whose graph describes this file.

    Nearest `graphify-out/` above it, so an automod worktree — where the
    round ran `graph_refresh` — gets its own graph rather than the live
    checkout's. Otherwise the live checkout, but only for a file actually
    inside it: answering about a tree the file is not in would name callers
    of somebody else's same-named symbol.
    """
    d = os.path.dirname(edited)
    while d and d != os.path.dirname(d):
        try:
            if cg.graph_path_for(Path(d)).is_file():
                return d
        except Exception:
            return ""
        d = os.path.dirname(d)
    if fallback_root and edited.startswith(fallback_root.rstrip("/") + "/"):
        try:
            if cg.graph_path_for(Path(fallback_root)).is_file():
                return fallback_root
        except Exception:
            return ""
    return ""


def _remember(root: str, entry: Any, cg: Any) -> None:
    try:
        st = cg.graph_path_for(Path(root)).stat()
        stamp = (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        stamp = (0, 0)
    with _GRAPH_LOCK:
        _GRAPHS[root] = (stamp[0], stamp[1], entry)
        _GRAPHS.move_to_end(root)
        while len(_GRAPHS) > MAX_CACHED_GRAPHS:
            _GRAPHS.popitem(last=False)


def _load_or_wait(cg: Any, root: str, deadline: float) -> Any:
    """Load `root`'s graph once; everyone else waits, bounded by the deadline.

    Two sessions race here for real — the aggregator serves them both — and
    the second must not pay a second 12 MB read. `threading.Event`, not
    `code_graph._lock_for`'s asyncio.Lock: this runs on worker threads, where
    an asyncio lock is either useless or fatal.
    """
    with _GRAPH_LOCK:
        if root in _MISS and time.perf_counter() < _MISS[root] + _MISS_TTL_S:
            return None
        ev = _LOADING.get(root)
        mine = ev is None
        if mine:
            _LOADING[root] = ev = threading.Event()
    if not mine:
        ev.wait(timeout=max(0.0, deadline - time.perf_counter()))
        with _GRAPH_LOCK:
            hit = _GRAPHS.get(root)
        return hit[2] if hit else None

    entry = None
    try:
        entry = cg._load_sync(Path(root))
        if entry is not None:
            _remember(root, entry, cg)
        else:
            with _GRAPH_LOCK:
                _MISS[root] = time.perf_counter()
    except Exception:
        logger.warning("blast-radius graph load failed for %s", root, exc_info=True)
        with _GRAPH_LOCK:
            _MISS[root] = time.perf_counter()
    finally:
        with _GRAPH_LOCK:
            _LOADING.pop(root, None)
        ev.set()
    return entry


def _entry_for(cg: Any, root: str, deadline: float) -> Any:
    """Cached graph for `root`, reloaded when graph.json has moved.

    A moved graph file reloads; if the reload cannot land inside the budget
    the *stale* entry is served, because a stale graph is the right picture
    of the pre-edit tree anyway. Only a missing one says nothing.
    """
    with _GRAPH_LOCK:
        hit = _GRAPHS.get(root)
    if hit is not None:
        try:
            st = cg.graph_path_for(Path(root)).stat()
        except OSError:
            st = None
        if st is not None and (int(st.st_mtime_ns), int(st.st_size)) == hit[:2]:
            with _GRAPH_LOCK:
                _GRAPHS.move_to_end(root)
            return hit[2]
    fresh = _load_or_wait(cg, root, deadline)
    return fresh or (hit[2] if hit else None)


def _node_here(cg: Any, entry: Any, sym: str, edited_rel: str) -> str:
    """Resolve `sym` and prove the node belongs to this file.

    `resolve_symbol`'s own file narrowing is a *preference*: when nothing
    matches the hint it falls back to every candidate. So the source_file
    check here is the one that decides, and it is what stops a same-named
    symbol in another module being reported as this edit's blast radius.
    """
    res = cg.resolve_symbol(entry, sym, edited_rel)
    node = res.node or ""
    if not node:
        return ""
    try:
        src = entry.graph.nodes[node].get("source_file") or ""
    except Exception:
        return ""
    return node if _same_file(src, edited_rel) else ""


def _call_sites(cg: Any, entry: Any, node: str, edited_rel: str) -> list[tuple[str, int]]:
    """Inbound callers outside the edited file, as (file, line) call sites.

    Depth 1 only: this is the edit's own reach, not a transitive closure,
    and a one-hop list is the longest one a model will read mid-edit.
    """
    g = entry.graph
    try:
        preds = cg._impact_preds(g, node)
    except Exception:
        return []
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for p in preds:
        try:
            attrs = g.nodes[p]
            rels = g[p][node]["rels"]
        except Exception:
            continue
        try:
            if cg._is_rationale(attrs):
                continue
        except Exception:                                 # pragma: no cover
            pass
        f = loc = ""
        for rel in rels:
            if rel.get("relation") not in cg.IMPACT_RELATIONS:
                continue
            if rel.get("source_file"):
                f, loc = rel["source_file"], rel.get("source_location") or ""
                break
        if not f:
            f = attrs.get("source_file") or ""
            loc = attrs.get("source_location") or ""
        if not f or _same_file(f, edited_rel):
            continue
        site = (f, _line_of(loc))
        if site not in seen:
            seen.add(site)
            out.append(site)
    return out


def _render_block(path: str, root: str, lines: list[str], nfiles: int,
                  suppressed: list[tuple[str, int]], omitted: int) -> str:
    n = len(lines)
    head = (f'<blast_radius file="{path}" root="{root}" tool="code_graph" '
            f'callers="{n}" files="{nfiles}" suppressed="{len(suppressed)}">')
    if n:
        summary = (f"advisory: {n} call site(s) in {nfiles} file(s) outside "
                   f"this file — check them against this edit")
    else:
        summary = "advisory: wider call sites exist outside this file but are too many to list"
    body = [head, summary, *lines]
    if omitted:
        body.append(f"... and {omitted} more call site(s) omitted "
                    f"(caps: {MAX_CALLER_FILES} files / {MAX_CALLER_LINES} lines)")
    if suppressed:
        named = ", ".join(f"`{s}` ({c})" for s, c in suppressed[:3])
        more = f" and {len(suppressed) - 3} more" if len(suppressed) > 3 else ""
        body.append(f"suppressed: {named}{more} — above the "
                    f"{FANOUT_CEILING}-call-site ceiling")
    return "\n".join(body) + "\n</blast_radius>"


def _probe_into(box: dict, path: str, real: str, fallback_root: str,
                touched: list[str], deadline: float) -> None:
    """Everything that touches the graph, in a thread the edit does not wait on.

    Catches its own exceptions because nothing up the stack can see them: the
    edit already returned by the time this raises, so an escaping exception
    becomes an unhandled-thread traceback in the log instead of a rail that
    quietly said nothing.
    """
    try:
        _probe(box, path, real, fallback_root, touched, deadline)
    except Exception:
        logger.warning("blast-radius probe failed for %s", path, exc_info=True)


def _probe(box: dict, path: str, real: str, fallback_root: str,
           touched: list[str], deadline: float) -> None:
    cg = _code_graph()
    if cg is None:
        return
    edited = os.path.realpath(real or path)
    root = _root_for(cg, edited, fallback_root)
    if not root:
        return
    edited_rel = _rel_to(root, edited)
    if not edited_rel:
        return
    entry = _entry_for(cg, root, deadline)
    if entry is None:
        return

    per_symbol: list[tuple[str, list[tuple[str, int]]]] = []
    suppressed: list[tuple[str, int]] = []
    for sym in touched:
        if time.perf_counter() > deadline:
            break
        node = _node_here(cg, entry, sym, edited_rel)
        if not node:
            continue
        sites = _call_sites(cg, entry, node, edited_rel)
        if not sites:
            continue
        if len(sites) > FANOUT_CEILING:
            suppressed.append((sym, len(sites)))
            continue
        per_symbol.append((sym, sites))

    flat = sorted((f, ln, sym)
                  for sym, sites in per_symbol for (f, ln) in sites)
    files: list[str] = []
    lines: list[str] = []
    omitted = 0
    for f, ln, sym in flat:
        if f not in files:
            if len(files) >= MAX_CALLER_FILES:
                omitted += 1
                continue
            files.append(f)
        if len(lines) >= MAX_CALLER_LINES:
            omitted += 1
            continue
        lines.append(f"{sym} \u2192 {f}:{ln}")

    if not lines and not suppressed:
        return
    box["block"] = _render_block(path, root, lines, len(files), suppressed, omitted)


def callers_block(path: str, pre_bytes: bytes | None, post_text: str, *,
                  real: str = "", fallback_root: str = "",
                  budget_s: float = RAIL_BUDGET_S) -> str:
    """The `<blast_radius>` block for one edit, or "" when there is nothing to say.

    Costs an edit at most `budget_s`: the work runs in a daemon thread the
    caller joins for that long and then abandons — and abandoning is not
    wasted, because the thread finishes the load and fills the per-root
    cache, so the *next* edit in the same tree gets its answer.
    """
    try:
        if not is_python(path):
            return ""
        if len(post_text.encode("utf-8", errors="replace")) > RAIL_MAX_SOURCE_BYTES:
            return ""
        if pre_bytes is None or len(pre_bytes) > RAIL_MAX_SOURCE_BYTES:
            return ""                 # a created file has no pre-edit callers
        touched = _touched_symbols(
            pre_bytes.decode("utf-8", errors="replace"), post_text)
        if not touched:
            return ""
        deadline = time.perf_counter() + max(0.001, budget_s)
        box: dict[str, str] = {}
        th = threading.Thread(
            target=_probe_into,
            args=(box, path, real, fallback_root, touched[:MAX_SYMBOLS], deadline),
            daemon=True, name="blast-radius")
        th.start()
        th.join(max(0.0, deadline - time.perf_counter()))
        return box.get("block", "")
    except Exception:
        logger.warning("blast-radius block failed for %s", path, exc_info=True)
        return ""


def config() -> dict:
    """`harness.edit_diagnostics`, read at call time so tests can patch it.

    `blast_radius` is the cross-file rail's own switch, separate from
    `python` because the two read different stores: pyflakes on the two
    images, and the code graph. Turning one off should not silently turn off
    the other. There is no `config.yaml` key for it yet — adding one is a
    human-only edit.
    """
    defaults = {"python": True, "typescript": True, "blast_radius": True,
                "max_lines": DEFAULT_MAX_LINES}
    try:
        from app.config import CONFIG
        raw = (CONFIG.get("harness") or {}).get("edit_diagnostics") or {}
    except Exception:
        raw = {}
    if isinstance(raw, dict):
        defaults.update({k: v for k, v in raw.items() if v is not None})
    return defaults
