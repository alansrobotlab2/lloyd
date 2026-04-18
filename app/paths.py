"""Shared filesystem paths for the Lloyd backend."""

from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parent.parent
SESSIONS_DIR = LLOYD_HOME / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

PIPELINE_RUNS_DIR = LLOYD_HOME / "pipeline-runs"
