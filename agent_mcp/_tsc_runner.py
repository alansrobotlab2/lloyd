"""Out-of-band TypeScript diagnostics for `.ts`/`.tsx` edits.

Python diagnostics run inline: pyflakes on one file is milliseconds, so the
result rides back on the Edit itself. tsc does not work that way. It is
whole-project only here — `web/tsconfig.json` has `include: ["src"]`, and
there is no per-file mode that resolves imports correctly — and a full
`tsc --noEmit -p .` measures ~5 s on this tree. Five seconds on every `.tsx`
edit is a tax on the common case to serve the rare one.

So it runs in the background and the answer arrives on a later iteration,
through the notification drain the background-Bash tool already uses. The
Edit result says a check was queued, which matters: a model told nothing
assumes nothing is coming, and a model told a check is running will not
re-run one itself.

Delta, like everything else here
--------------------------------
The tree carries pre-existing tsc errors, and the gate's frontend rung
judges them as a delta for exactly that reason. A whole-project run is
turned into a per-session answer by keeping the previous run's per-file
counts as a baseline and reporting only `run[f] - baseline[f]` for the files
*that session* edited. Another session's breakage is that session's news.

**Cold start is the trap.** With no baseline, the first run after a restart
would attribute every pre-existing error in the tree to whoever edited first.
So the first run only seeds, and `warm_baseline()` is scheduled from the
aggregator's lifespan a few seconds after boot — otherwise the first `.tsx`
edit of every restart is the one that gets no answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

from app.lint_findings import node_env, parse_tsc, split_tsc_by_file

logger = logging.getLogger("lloyd-tsc-runner")

SUFFIXES = (".ts", ".tsx")
DEBOUNCE_S = 1.5
TIMEOUT_S = 120.0
MAX_REPORTED_LINES = 30

HINT = ("(tsc check queued; new TypeScript errors, if any, arrive as a "
        "<diagnostics_notification> on a later iteration)")

# One tsc at a time in this process, whatever the root. Two concurrent
# whole-project type-checks on the same box are slower than one after the
# other and would interleave their baselines.
_LOCK = asyncio.Lock()

# root -> session_id -> set of paths relative to <root>/web
_pending: dict[str, dict[str, set[str]]] = {}
# root -> debounce task
_timers: dict[str, asyncio.Task] = {}
# root -> file -> Counter of normalised findings
_baseline: dict[str, dict[str, Counter]] = {}
# root -> last run summary, for /state
_last_run: dict[str, dict] = {}
_current: object | None = None


def _cfg() -> dict:
    try:
        from agent_mcp._edit_diagnostics import config
        return config()
    except Exception:
        return {"typescript": True, "max_lines": MAX_REPORTED_LINES}


def project_root(real_path: str) -> Path | None:
    """The tree `real_path` belongs to, or None if tsc cannot say anything.

    Walks up from the file looking for a `web/tsconfig.json`, so a autoimplement
    worktree answers about itself rather than about the live checkout. The
    file must be under that project's `web/src`, which is the only thing
    `tsconfig.json` includes.
    """
    if not real_path.endswith(SUFFIXES):
        return None
    p = Path(real_path)
    for parent in p.parents:
        if (parent / "web" / "tsconfig.json").is_file():
            try:
                p.relative_to(parent / "web" / "src")
            except ValueError:
                return None
            return parent
    return None


def _tsc_bin(root: Path) -> Path | None:
    """`node_modules` is gitignored, so a fresh worktree has none until the
    gate symlinks the live one in. No binary means no run and no hint —
    promising a check that cannot happen is worse than saying nothing."""
    b = root / "web" / "node_modules" / ".bin" / "tsc"
    return b if b.exists() else None


def _baseline_path(root: Path) -> Path:
    return root / "_pipeline" / "tsc" / "baseline.json"


def _load_baseline(root: Path) -> dict[str, Counter] | None:
    key = str(root)
    if key in _baseline:
        return _baseline[key]
    try:
        raw = json.loads(_baseline_path(root).read_text())
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    loaded = {f: Counter(c) for f, c in raw.items() if isinstance(c, dict)}
    _baseline[key] = loaded
    return loaded


def _save_baseline(root: Path, by_file: dict[str, Counter]) -> None:
    _baseline[str(root)] = by_file
    path = _baseline_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({f: dict(c) for f, c in by_file.items()}))
        os.replace(tmp, path)
    except Exception:
        logger.warning("tsc: could not persist baseline at %s", path, exc_info=True)


async def note_edit(session_id: str, real_path: str) -> str:
    """Record a `.ts`/`.tsx` edit and (re)arm the debounce. Returns the hint."""
    if not _cfg().get("typescript", True):
        return ""
    root = project_root(real_path)
    if root is None or _tsc_bin(root) is None:
        return ""
    try:
        rel = str(Path(real_path).relative_to(root / "web"))
    except ValueError:
        return ""

    key = str(root)
    _pending.setdefault(key, {}).setdefault(session_id or "", set()).add(rel)

    timer = _timers.get(key)
    if timer is not None and not timer.done():
        timer.cancel()
    _timers[key] = asyncio.create_task(_debounced(root), name=f"tsc-debounce-{key}")
    return HINT


async def _debounced(root: Path) -> None:
    try:
        await asyncio.sleep(DEBOUNCE_S)
    except asyncio.CancelledError:
        return
    try:
        await run_once(root)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("tsc: background run failed", exc_info=True)


async def _spawn_tsc(root: Path) -> tuple[str, str, float]:
    """Run the type-check. Returns (status, output, seconds)."""
    global _current
    bin_path = _tsc_bin(root)
    if bin_path is None:
        return "failed", "no tsc binary under web/node_modules/.bin", 0.0
    web = root / "web"
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            str(bin_path), "--noEmit", "-p", ".",
            cwd=str(web), env=node_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        return "failed", f"could not start tsc: {exc}", 0.0
    _current = proc
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_S)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        from agent_mcp.builtin_bash import _kill_proc_tree
        _kill_proc_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        _current = None
        return "timeout", f"tsc did not finish within {TIMEOUT_S:.0f}s", \
            time.monotonic() - started
    finally:
        if _current is proc:
            _current = None
    text = out.decode("utf-8", errors="replace") + err.decode("utf-8", errors="replace")
    return "ok", text, time.monotonic() - started


async def run_once(root: Path, *, seed_only: bool = False) -> dict:
    """One type-check, coalescing every session's pending files for `root`."""
    key = str(root)
    async with _LOCK:
        pending = _pending.pop(key, {})
        if not pending and not seed_only:
            return {"skipped": "nothing pending"}

        had_baseline = _load_baseline(root) is not None
        status, text, seconds = await _spawn_tsc(root)
        summary = {
            "root": key, "status": status, "seconds": round(seconds, 1),
            "at": time.time(),
            "sessions": sorted(pending),
        }

        if status != "ok":
            summary["detail"] = text[:300]
            _last_run[key] = summary
            for sid, files in pending.items():
                await _emit(sid, sorted(files), [], status, text[:300],
                            seconds, root)
            return summary

        counter = parse_tsc(text)
        by_file = split_tsc_by_file(counter)
        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        summary["errors_total"] = sum(counter.values())
        base = _load_baseline(root) or {}
        _save_baseline(root, by_file)
        _last_run[key] = summary

        if not had_baseline or seed_only:
            # First run since boot: seed only. Attributing every pre-existing
            # error in the tree to whoever edited first is the one way to make
            # this feature actively misleading.
            summary["seeded"] = True
            return summary

        for sid, files in pending.items():
            new: Counter = Counter()
            for f in files:
                new += by_file.get(f, Counter()) - base.get(f, Counter())
            if not new:
                continue
            lines = _display_lines(raw_lines, new,
                                   int(_cfg().get("max_lines", MAX_REPORTED_LINES)))
            await _emit(sid, sorted(files), lines, "ok", "", seconds, root)
        return summary


def _display_lines(raw_lines: list[str], new: Counter, max_lines: int) -> list[str]:
    """Raw tsc lines (with line:col) for the findings the delta says are new.

    The delta is computed on normalised findings — position dropped, so a
    file that grew ten lines does not report everything below as new — but
    what the model needs to act on is the position. Same shape as the
    pyflakes block.
    """
    remaining = Counter(new)
    out: list[str] = []
    for line in raw_lines:
        norm = parse_tsc(line)
        for finding in norm:
            if remaining.get(finding, 0) > 0:
                remaining[finding] -= 1
                out.append(line)
    total = len(out)
    if max_lines > 0 and total > max_lines:
        out = out[:max_lines] + [f"... and {total - max_lines} more"]
    return out


def _notify_target(session_id: str) -> str:
    """Where a late result should land.

    A subagent's drain queue is never read after its Task returns, so a
    `task:*` session is redirected to the parent that spawned it.
    """
    if not session_id.startswith("task:"):
        return session_id
    try:
        from agent_mcp import _subagent_registry
        scope = _subagent_registry.parent_scope(session_id)
    except Exception:
        scope = None
    return scope[0] if scope else session_id


async def _emit(session_id: str, files: list[str], lines: list[str],
                status: str, detail: str, seconds: float, root: Path) -> None:
    target = _notify_target(session_id)
    if not target:
        return
    from agent_mcp import _task_registry
    now = time.time()
    await _task_registry.enqueue_diagnostics(_task_registry.DiagnosticsRecord(
        session_id=target,
        kind="typescript",
        files=files,
        lines=lines,
        started_at=now - seconds,
        finished_at=now,
        status=status,
        detail=detail,
    ))


async def warm_baseline(delay_s: float = 20.0) -> None:
    """Seed the baseline shortly after boot, so the first edit still deltas."""
    try:
        await asyncio.sleep(delay_s)
    except asyncio.CancelledError:
        return
    if not _cfg().get("typescript", True):
        return
    from app.paths import LLOYD_HOME
    root = Path(os.path.realpath(LLOYD_HOME))
    if _tsc_bin(root) is None:
        return
    try:
        await run_once(root, seed_only=True)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("tsc: baseline warm-up failed", exc_info=True)


async def shutdown() -> None:
    for task in list(_timers.values()):
        if not task.done():
            task.cancel()
    _timers.clear()
    proc = _current
    if proc is not None and getattr(proc, "returncode", 0) is None:
        try:
            from agent_mcp.builtin_bash import _kill_proc_tree
            _kill_proc_tree(proc)
        except Exception:
            pass


def stats() -> dict:
    return {
        "last_run": dict(_last_run),
        "pending": {r: {s: sorted(f) for s, f in by_sid.items()}
                    for r, by_sid in _pending.items()},
        "baseline_roots": sorted(_baseline),
    }


def reset() -> None:
    """Tests only."""
    _pending.clear()
    for task in list(_timers.values()):
        task.cancel()
    _timers.clear()
    _baseline.clear()
    _last_run.clear()
