"""`data/tool_overrides.yaml` — the UI-mutable slice, and why it is not source.

Two properties, and they are the same property seen from two sides.

The Tools page must be able to change tool state without editing
`config.yaml`, because `config.yaml` is tracked and a tracked file rewritten
by a UI click leaves the live tree dirty — which `scripts/selfmod/gate.py`
and `promote.py` both refuse. That is why `save_tool_overrides` exists at
all; its docstring records that it replaced dumping the whole CONFIG back
over `config.yaml` on every toggle.

The override file was itself tracked until 2026-09-07, so the escape hatch
had the defect it was built to avoid: one toggle in Mission Control dirtied
the tree and silently stopped the self-modification loop until a human
committed the result. `git log data/tool_overrides.yaml` shows exactly that
happening — UI state hand-committed as if it were source.

The other side of it: once the override wins silently, the tracked file is
free to describe a state nobody is serving. `config.yaml` claimed
`tool_search.enabled: true` while the override served `false`, and nothing
logged the disagreement, so the only way to learn how tools were actually
advertised was to evaluate the merge by hand.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from app.config import _merge_tool_overrides  # noqa: E402

OVERRIDES = "data/tool_overrides.yaml"


# ---------------------------------------------------------------------------
# Repo hygiene — a toggle must not be able to dirty the tree
# ---------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    )


def test_the_override_file_is_not_tracked():
    """Tracking it re-arms the bug the file exists to prevent.

    `worktree.is_clean` is a bare `git status --porcelain`, so a modified
    tracked file and an unignored new file both read as dirty. The gate and
    the promoter each refuse a dirty live tree, which means re-adding this
    file turns every Tools-page click into a silent outage of the
    self-modification loop.
    """
    r = _git("ls-files", "--error-unmatch", OVERRIDES)
    assert r.returncode != 0, (
        f"{OVERRIDES} is tracked again. The Tools page rewrites it on every "
        f"toggle, so tracking it means a UI click leaves `git status` dirty "
        f"and the selfmod gate/promoter refuse to run."
    )


def test_the_override_file_is_ignored():
    """Untracked is not enough — `git status --porcelain` lists new files too.

    Without a matching .gitignore rule the file shows up as `??` and the
    tree is dirty for the same reason, just through a different column.
    """
    assert _git("check-ignore", "-q", OVERRIDES).returncode == 0, (
        f"{OVERRIDES} is untracked but not ignored, so it still shows in "
        f"`git status --porcelain` and still dirties the live tree"
    )


def test_a_written_override_leaves_the_live_tree_clean():
    """The end-to-end property the other two exist to guarantee."""
    assert (ROOT / OVERRIDES).exists(), "expected the live override file present"
    assert not _git("status", "--porcelain", OVERRIDES).stdout.strip(), (
        "writing the override file must not register in `git status` at all"
    )


# ---------------------------------------------------------------------------
# A silent win hides a real disagreement
# ---------------------------------------------------------------------------


def _config(**tool_search) -> dict:
    return {"mcp_servers": {}, "harness": {"tool_search": dict(tool_search)}}


def test_a_shadowed_tool_search_key_is_logged(tmp_path, monkeypatch, caplog):
    """`enabled` decides whether the model sees 131 tools or a baseline.

    It is about as load-bearing as a flag gets, and it was merged with a
    bare `.update()` — so config.yaml said `true`, the override served
    `false`, and the log said nothing in either direction.
    """
    ovr = tmp_path / "tool_overrides.yaml"
    ovr.write_text("harness:\n  tool_search:\n    enabled: false\n")
    monkeypatch.setattr("app.config.TOOL_OVERRIDES_PATH", ovr)

    with caplog.at_level(logging.WARNING, logger="lloyd-config"):
        merged = _merge_tool_overrides(_config(enabled=True, threshold_tools=30))

    assert merged["harness"]["tool_search"]["enabled"] is False, "override still wins"
    text = caplog.text
    assert "tool_search" in text and "enabled" in text, (
        f"a shadowed tool_search key must be logged; got: {text!r}"
    )


def test_agreeing_files_log_nothing(tmp_path, monkeypatch, caplog):
    """The warning is a disagreement signal, not merge noise.

    An override that restates the tracked value is the normal steady state —
    the Tools page writes the whole block back every time — so warning on it
    would fire on every boot and stop meaning anything.
    """
    ovr = tmp_path / "tool_overrides.yaml"
    ovr.write_text("harness:\n  tool_search:\n    enabled: false\n")
    monkeypatch.setattr("app.config.TOOL_OVERRIDES_PATH", ovr)

    with caplog.at_level(logging.WARNING, logger="lloyd-config"):
        _merge_tool_overrides(_config(enabled=False, threshold_tools=30))

    assert "tool_search" not in caplog.text, (
        f"no disagreement, so nothing to report; got: {caplog.text!r}"
    )


def test_a_key_absent_from_config_is_not_a_disagreement(tmp_path, monkeypatch, caplog):
    """An override may carry keys config.yaml never mentions."""
    ovr = tmp_path / "tool_overrides.yaml"
    ovr.write_text("harness:\n  tool_search:\n    max_results_cap: 20\n")
    monkeypatch.setattr("app.config.TOOL_OVERRIDES_PATH", ovr)

    with caplog.at_level(logging.WARNING, logger="lloyd-config"):
        merged = _merge_tool_overrides(_config(enabled=False))

    assert merged["harness"]["tool_search"]["max_results_cap"] == 20
    assert "tool_search" not in caplog.text


# ---------------------------------------------------------------------------
# The tracked defaults must describe what a rebuild would actually serve
# ---------------------------------------------------------------------------


def test_config_yaml_agrees_with_the_live_override():
    """Now that the override is gitignored, config.yaml is the rebuild path.

    SETUP.md tells you to restore this file from backup and warns that
    without it tool state falls back to config.yaml's defaults. So a
    disagreement is no longer just confusing documentation — it is the state
    a fresh clone would boot into.
    """
    import yaml

    live = ROOT / OVERRIDES
    if not live.exists():  # a clean checkout; nothing to reconcile against
        return
    ovr = (yaml.safe_load(live.read_text()) or {}).get("harness", {}).get(
        "tool_search", {}
    )
    tracked = (
        (yaml.safe_load((ROOT / "config.yaml").read_text()) or {})
        .get("harness", {})
        .get("tool_search", {})
    )
    drift = {
        k: (tracked.get(k), v)
        for k, v in (ovr or {}).items()
        if k in tracked and tracked[k] != v
    }
    assert not drift, (
        f"config.yaml disagrees with the served override on {drift} — a fresh "
        f"clone would boot into the tracked value, not this one"
    )
