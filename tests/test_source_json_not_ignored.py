"""Hand-authored JSON under `scripts/` and `app/` is source, not runtime state (#2115).

`.gitignore` carries a repo-wide `*.json` (runtime state and logs that may hold
personal data). Until 2026-10-04 it also hid any new schema or fixture written
in the two source trees: `git add -A` skipped it without a word, so a round
could land code reading a file that was never committed. Two negations re-include
those trees; these pin both, and that the blanket rule still stands everywhere else.

Two boundaries that were decided in #2115 but left unpinned are pinned here
(#2321). First, a carve-out reaches only into directories git descends into: JSON
under a parent the file excludes as a DIRECTORY (`.venv/`, `.venvs/`) stays hidden
even inside `scripts/` and `app/`, because a negation cannot re-include a path
whose parent was never walked. Second, nothing here stops a `node_modules`
appearing under `scripts/` or `app/`, where the sole node_modules rule does not
reach it and the carve-out would re-include every JSON inside it.
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
# Path -> the `.gitignore` rule that hides it today, read off
# `git check-ignore -v --no-index <path>`. The assert names that rule, so the
# case fails with its culprit when the line is deleted or narrowed.
STILL_IGNORED = {
    "stray.json": "*.json",
    "sessions/20261004_000000_x.json": "/sessions/",
    "agent_mcp/state.json": "*.json",
    "workers/dump.json": "*.json",
    "chrome-extension/not-a-manifest.json": "*.json",
    # JSON under a directory-excluded parent, inside both source carve-outs. Git
    # never descends into an excluded directory, so the later
    # `!scripts/**/*.json` / `!app/**/*.json` cannot re-include what sits under
    # one — these hold on `.venv/`/`.venvs/`, not on the dot in the name:
    # `scripts/.cache/lib/state.json` is covered by no such rule and is NOT
    # ignored (it last-matches `!scripts/**/*.json`). Both unanchored venv
    # spellings bite inside both trees, so both are pinned.
    "scripts/.venvs/lib/state.json": ".venvs/",
    "scripts/.venv/lib/state.json": ".venv/",
    "app/.venvs/x/state.json": ".venvs/",
}

# The two source trees whose carve-out would swallow a dependency tree (#2115
# owed entry 2, measured then and re-measured at #2321: neither exists).
NODE_FREE_SOURCE_DIRS = ("scripts/node_modules", "app/node_modules")


def _ignored(rel_path: str) -> int:
    # `-q` exits 0 when the path IS ignored, 1 when it is not, 128 on error.
    # Never add `-v` to an assertion: it prints the matching rule but exits 0 for
    # a re-included path too, so it reports a negated path as ignored.
    return subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", "--no-index", rel_path],
                          capture_output=True, text=True).returncode


@pytest.mark.parametrize("rel_path", SOURCE_JSON)
def test_a_new_source_json_would_be_seen_by_git_add(rel_path: str):
    assert _ignored(rel_path) == 1, f"{rel_path} is ignored: `git add -A` would skip it silently"


@pytest.mark.parametrize(
    "rel_path,rule",
    [pytest.param(rel_path, rule, id=rel_path) for rel_path, rule in STILL_IGNORED.items()])
def test_the_blanket_rule_still_hides_json_everywhere_else(rel_path: str, rule: str):
    assert _ignored(rel_path) == 0, (
        f"{rel_path} is no longer ignored: the `{rule}` rule that hid it was deleted "
        f"or negated, so `git add -A` would now stage it")


def test_the_blanket_rule_itself_is_still_in_the_file():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*.json" in lines
    assert "!scripts/**/*.json" in lines and "!app/**/*.json" in lines
    assert lines.index("*.json") < lines.index("!scripts/**/*.json")


def test_the_repo_the_tripwire_reads_is_a_real_checkout():
    """Positive control for the tripwire below. `REPO` is resolved from this
    file, so it is the checkout the test lives in — including a
    `git worktree add --detach` one that `scripts/automod/gate.py` may have
    symlinked `web/node_modules` into. If it were not a populated tree, both
    `.exists()` checks below would return False and the tripwire would pass
    without having looked at anything."""
    assert (REPO / ".gitignore").exists(), f"{REPO} is not a Lloyd checkout root"
    assert (REPO / "web" / "package.json").exists(), f"{REPO} has no web/ tree in it"


def test_no_node_modules_has_appeared_under_a_source_carve_out():
    # Exactly these two paths, by `.exists()`, not a recursive scan: two
    # node_modules DO belong on this box and a walk would trip on them —
    # `web/node_modules`, which `scripts/automod/gate.py` symlinks into every
    # worktree it type-checks, and `qmd/node_modules`, excluded wholesale by the
    # `/qmd/` rule. Neither is named here.
    for rel_path in NODE_FREE_SOURCE_DIRS:
        assert not (REPO / rel_path).exists(), (
            f"{rel_path} exists, and no `.gitignore` line reaches it: the only "
            f"node_modules rule is the anchored `/web/node_modules`, so the "
            f"`!{rel_path.split('/')[0]}/**/*.json` carve-out would re-include every "
            f"JSON inside it and `git add -A` would stage a dependency tree. Add the "
            f"anchored lines `/scripts/node_modules` and `/app/node_modules` to "
            f"`.gitignore` (a human-only edit — this test only names them), or remove "
            f"the directory.")
