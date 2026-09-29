"""The Chrome side-panel extension's manifest must be in the tree (#1701).

A repo-wide `*.json` rule in `.gitignore` hid `chrome-extension/manifest.json`
while `chrome-extension/dist/**` (service worker, side panel, assets, icons) was
tracked, so a fresh clone shipped a `dist/` Chrome cannot load, and nothing
could regenerate it: `web/vite.chrome.config.ts` copies the manifest, it does
not generate it. `git add -A` skips an ignored file silently, which is how it
stayed out for two weeks.

Both copies are pinned: the hand-written source, and the one the build writes
into the tracked `dist/` — keeping `dist/` tracked without its manifest would
leave the committed extension one file short again.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

MANIFESTS = (
    "chrome-extension/manifest.json",
    "chrome-extension/dist/manifest.json",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


@pytest.mark.parametrize("rel_path", MANIFESTS)
def test_manifest_is_tracked(rel_path: str):
    res = _git("ls-files", "--error-unmatch", rel_path)
    assert res.returncode == 0, (
        f"{rel_path} is not tracked — a fresh clone gets an unloadable extension: "
        f"{res.stderr.strip()}")


@pytest.mark.parametrize("rel_path", MANIFESTS)
def test_manifest_is_not_ignored(rel_path: str):
    # `-q` exits 0 when the path IS ignored, 1 when it is not (a negation that
    # re-includes it counts as not ignored), 128 on error.
    res = _git("check-ignore", "-q", "--no-index", rel_path)
    assert res.returncode == 1, (
        f"{rel_path} is ignored (exit {res.returncode}) — `git add -A` would skip "
        f"the next change to it silently: {_git('check-ignore', '-v', '--no-index', rel_path).stdout.strip()}")


def test_the_ignore_check_can_fail():
    """Positive control: the blanket `*.json` rule still hides an ordinary JSON
    file beside the manifest, so the exit-1 above means a negation matched,
    not that `check-ignore` stopped answering."""
    res = _git("check-ignore", "-q", "--no-index", "chrome-extension/not-a-manifest.json")
    assert res.returncode == 0, "the *.json rule no longer hides stray JSON — the control is inert"
