"""No code resolves a runtime-data path off the code tree.

Since 2026-09-22 runtime data lives in `~/lloyd-data` (`architecture/data-home.md`)
and every writer reaches it through `app.paths.DATA_ROOT`. Before the move, some
60 sites built their own path — `Path.home()/"lloyd"/"sessions"`,
`Path(__file__).parent/"usage.db"`, `~/lloyd/workers.db` in config — and one
missed site is enough to start a second, silent copy of the data inside the tree
that every delete aimed at the code reaches again. The guardian names such a
copy an hour after it appears (`datawatch.stray_in_tree`); this names the code
that would write it, before it lands.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNTIME = (r"(?:sessions|event_logs|_pipeline|autonomy-runs|logs|usage\.db|workers\.db|"
           r"research\.db|mc-state\.json|voice_profiles|agent-services/logs|eval/baselines|"
           r"data/tool_overrides\.yaml)")
_Q = r"""["']"""
PATTERNS = [
    # ~/lloyd/X, $HOME/lloyd/X, /home/<user>/lloyd/X, %h/lloyd/X
    re.compile(r"(?:~|\$HOME|\$\{HOME\}|%h|/home/[a-z_][a-z0-9_-]*)/lloyd/" + RUNTIME + r"(?![\w.-])"),
    # Path.home() / "lloyd" / "X"
    re.compile(r"Path\.home\(\)\s*/\s*" + _Q + r"lloyd" + _Q + r"\s*/\s*" + _Q + RUNTIME + _Q),
    # LLOYD_HOME / "X", LIVE_CHECKOUT / "X", REPO / "X"
    re.compile(r"\b(?:LLOYD_HOME|LIVE_CHECKOUT|LIVE_ROOT|_LLOYD_ROOT|_LLOYD_HOME)\s*/\s*"
               + _Q + RUNTIME + _Q),
    # Path(__file__)…parent / "usage.db" etc.
    re.compile(r"__file__\)[\w.()\[\]]*\s*/\s*" + _Q + RUNTIME + _Q),
]

SUFFIXES = {".py", ".sh", ".conf", ".yaml", ".yml", ".service", ".timer", ".ts", ".tsx"}
#: Files whose job is naming the old layout.
ALLOWED = {
    "scripts/migrate_data_home.py",
    "scripts/maintenance/rewrite_vault_data_paths.py",
    "scripts/maintenance/cutover_data_home.sh",
    "tests/test_no_runtime_paths_in_code.py",
}
#: Dated measurement scripts are a record of how a number was taken, not code
#: that runs; tests carry the old layout as fixture text.
ALLOWED_PREFIXES = ("tests/", "eval/measurements/", "agent-services/llm/")


def _tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                         text=True, check=True).stdout.split("\n")
    return [f for f in out if f and Path(f).suffix in SUFFIXES
            and f not in ALLOWED and not f.startswith(ALLOWED_PREFIXES)]


def test_no_tracked_code_builds_a_runtime_path_off_the_tree():
    hits = []
    for rel in _tracked():
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(p.search(line) for p in PATTERNS):
                hits.append(f"{rel}:{n}: {line.strip()[:140]}")
    assert not hits, ("runtime data resolved off the code tree — use app.paths "
                      "(DATA_ROOT and its constants) or ${LLOYD_DATA} in config:\n  "
                      + "\n  ".join(hits))


def test_the_patterns_catch_every_spelling_that_shipped():
    shipped = [
        'DB_PATH = Path(__file__).parent / "usage.db"',
        '_STATE_PATH = LLOYD_HOME / "mc-state.json"',
        'SESSIONS_DIR = Path.home() / "lloyd" / "sessions"',
        'raw = cfg.get("db_path") or "~/lloyd/workers.db"',
        'stdout_logfile=/home/alansrobotlab/lloyd/logs/server.log',
        'SESSIONS="$HOME/lloyd/_pipeline/vault-derived/sessions"',
        "ExecStart=%h/lloyd/logs/x",
    ]
    for line in shipped:
        assert any(p.search(line) for p in PATTERNS), line
    for fine in ('Path.home() / "lloyd" / "scripts"', "~/lloyd/.venvs/lloyd/bin/python",
                 "~/lloyd-data/sessions", 'LLOYD_HOME / "config.yaml"'):
        assert not any(p.search(fine) for p in PATTERNS), fine
