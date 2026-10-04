"""Hand-authored JSON under `scripts/` and `app/` is source, not runtime state (#2115).

`.gitignore` carries a repo-wide `*.json` (runtime state and logs that may hold
personal data). Until 2026-10-04 it also hid any new schema or fixture written
in the two source trees: `git add -A` skipped it without a word, so a round
could land code reading a file that was never committed. Two negations re-include
those trees; these pin both, and that the blanket rule still stands everywhere else.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SOURCE_JSON = [
    "scripts/memory/new_thing_schema.json",
    "scripts/new_top_level.json",
    "scripts/a/b/c/deep.json",
    "app/harness/new_schema.json",
    "app/new_top_level.json",
]
STILL_IGNORED = [
    "stray.json",
    "sessions/20261004_000000_x.json",
    "agent_mcp/state.json",
    "workers/dump.json",
    "chrome-extension/not-a-manifest.json",
]


def _ignored(rel_path: str) -> int:
    # `-q` exits 0 when the path IS ignored, 1 when it is not, 128 on error.
    return subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", "--no-index", rel_path],
                          capture_output=True, text=True).returncode


@pytest.mark.parametrize("rel_path", SOURCE_JSON)
def test_a_new_source_json_would_be_seen_by_git_add(rel_path: str):
    assert _ignored(rel_path) == 1, f"{rel_path} is ignored: `git add -A` would skip it silently"


@pytest.mark.parametrize("rel_path", STILL_IGNORED)
def test_the_blanket_rule_still_hides_json_everywhere_else(rel_path: str):
    assert _ignored(rel_path) == 0, f"{rel_path} is no longer ignored: the *.json guard was widened"


def test_the_blanket_rule_itself_is_still_in_the_file():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*.json" in lines
    assert "!scripts/**/*.json" in lines and "!app/**/*.json" in lines
    assert lines.index("*.json") < lines.index("!scripts/**/*.json")
