"""One qmd on the machine: the fork in ~/lloyd/qmd.

Until 2026-09-19 the daemon served the index from the fork while the watcher,
the index-maintenance task and both setup scripts ran the published
`@tobilu/qmd` from bun's global install -- same version string (2.8.3),
different commit. Nothing had broken, because the fork's changes are
serve-side; what existed was a second definition of how the index gets built,
and a `qmd` on PATH that was not the one answering queries. These tests are the
reason a fifth caller written next month cannot bring that back quietly.

Nothing here needs the fork to be present: `qmd/` is a gitignored clone and is
absent from every automod worktree, so the pins read tracked text only.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORK_JS = "lloyd/qmd/dist/cli/qmd.js"

DAEMON_CONF = ROOT / "agent-services/supervisor/conf.d/agent-qmd-daemon.conf"
WATCHER = ROOT / "agent-services/scripts/qmd-watcher.sh"
CLEANUP_UNIT = ROOT / "agent-services/systemd/lloyd-qmd-cleanup.service"
MAINTENANCE = ROOT / "scripts/maintenance/qmd_index_maintenance.py"


def _code_lines(path: Path) -> list[str]:
    """Lines that execute: comments carry this file's history on purpose."""
    return [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not ln.lstrip().startswith(("#", ";"))]


def _line(path: Path, pattern: str) -> str:
    hits = [ln for ln in _code_lines(path) if re.search(pattern, ln)]
    assert len(hits) == 1, f"{path.name}: expected exactly one line matching {pattern!r}, got {hits}"
    return hits[0]


def test_every_caller_runs_the_forks_cli():
    callers = {
        DAEMON_CONF: r"^command=",
        WATCHER: r"^QMD_CLI=",
        CLEANUP_UNIT: r"^ExecStart=",
        MAINTENANCE: r"^QMD_CLI\s*=",
    }
    for path, pattern in callers.items():
        line = _line(path, pattern)
        assert FORK_JS in line.replace("%h/", "lloyd/../").replace("$HOME/", "") or \
            "qmd/dist/cli/qmd.js" in line and ".bun" not in line, (
            f"{path.relative_to(ROOT)} does not run the fork: {line.strip()}")
        assert ".bun" not in line and "node_modules/@tobilu" not in line, (
            f"{path.relative_to(ROOT)} runs the published build: {line.strip()}")


def test_no_tracked_code_reaches_the_published_build():
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split("\0")
    suffixes = {".py", ".sh", ".conf", ".service", ".timer", ".yaml", ".yml", ".ts", ".tsx"}
    bad = re.compile(r"\.bun/bin/qmd|node_modules/@tobilu/qmd|bun (install|add) -g @tobilu/qmd")
    offenders = []
    for rel in tracked:
        path = ROOT / rel
        if not rel or path.suffix not in suffixes or not path.is_file():
            continue
        if path == Path(__file__).resolve():
            continue
        for n, ln in enumerate(_code_lines(path), 1):
            if bad.search(ln):
                offenders.append(f"{rel}: {ln.strip()[:120]}")
    assert not offenders, "the published qmd is back:\n  " + "\n  ".join(offenders)


def test_the_maintenance_task_never_takes_retrieval_down():
    """It stopped the daemon on eight of its last eight runs to embed one to four
    documents: pending embeddings are never zero while the loop is writing, and
    the stop was a requirement of the published build's cleanup, not the fork's."""
    src = "\n".join(_code_lines(MAINTENANCE))
    assert 'supervisor("stop")' not in src
    assert 'supervisor("restart")' in src, "an unhealthy daemon must still be brought back"


def test_the_watcher_defines_its_helpers_before_the_pipeline():
    """`bash -n` passes with a function definition sitting between `inotifywait |`
    and `while read` -- and the loop then reads the script's stdin instead of
    the event stream. The first cut of the throttle shipped exactly that."""
    text = WATCHER.read_text()
    pipe = text.index("inotifywait -m")
    for helper in ("debounce()", "drain_for()"):
        assert text.index(helper) < pipe, f"{helper} must be defined above the pipeline"
    tail = text[pipe:]
    assert re.search(r"\|\s*\nwhile read -r; do", tail), "the loop is no longer the pipe's reader"


def test_the_watcher_throttles_a_continuous_event_stream(tmp_path):
    """A no-op `qmd update` re-hashes every file in every collection (7.5 s for
    15,778 files, 2026-09-19) and forces the daemon's in-memory vector index to
    rebuild on the next query. The debounce alone let one run per write."""
    calls = tmp_path / "calls"
    stub = tmp_path / "qmd-stub"
    stub.write_text(f"#!/bin/bash\necho \"$2\" >> {calls}\n")
    stub.chmod(0o755)
    text = WATCHER.read_text()
    text = re.sub(r"^QMD_CLI=.*$", f'QMD_CLI=("{stub}" "x")', text, flags=re.M)
    text = re.sub(r"^COOLDOWN_SEC=.*$", "COOLDOWN_SEC=3", text, flags=re.M)
    text = re.sub(r"^MAX_DEBOUNCE_SEC=.*$", "MAX_DEBOUNCE_SEC=1", text, flags=re.M)
    text = re.sub(r"^DEBOUNCE_SEC=.*$", "DEBOUNCE_SEC=1", text, flags=re.M)
    text, n = re.subn(r"^inotifywait -m .*\\\n.*\|\s*$",
                      "(while true; do echo ev.md; sleep 0.1; done) |", text, flags=re.M)
    assert n == 1, "could not swap the event source; the pipeline's shape changed"
    text = text.replace('mkdir -p "$SESSIONS"', ":")
    script = tmp_path / "watcher.sh"
    script.write_text(text)
    try:
        subprocess.run(["bash", str(script)], timeout=7, capture_output=True)
    except subprocess.TimeoutExpired:
        pass
    updates = calls.read_text().split().count("update") if calls.exists() else 0
    # 7 s of events every 0.1 s: unthrottled that is a cycle per debounce (5+);
    # with a 3 s floor it is two, three if the clock is unkind.
    assert 1 <= updates <= 3, f"{updates} index walks in 7 s under a continuous stream"
