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

import contextlib
import logging
import os
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
    """The end-to-end property the other two exist to guarantee.

    This *writes* the file, through the real writer, into whatever tree the
    suite is running in, and asserts git never sees it. It used to assert
    only that the live file was already `.exists()` — which is not a property
    of this repo at all: `data/` has no tracked contents, so a fresh worktree
    contains no `data/` and the assertion failed for every self-modification
    round from `d11ad8c` onward, whatever the diff under test. For fifteen
    hours the loop's only drain was blocked by a test asserting that a
    deliberately-untracked file had been checked out, and three rounds
    aborted on it while filing fourteen new backlog items about other things.
    A test about untracked state may not require that state to be present.

    Writing is also the stronger check. The old assertion could only observe
    a file somebody else had already written and reverted; it could not have
    caught a writer that dirtied the tree, which is the entire failure this
    module exists to prevent. `app.paths.LLOYD_HOME` resolves from `__file__`,
    so the writer and this test address the same tree in a worktree too —
    asserted below rather than assumed, because that is the coupling that
    makes writing here safe.
    """
    from app.config import TOOL_OVERRIDES_PATH, save_tool_overrides

    assert TOOL_OVERRIDES_PATH == ROOT / OVERRIDES, (
        f"the writer targets {TOOL_OVERRIDES_PATH} but this test guards "
        f"{ROOT / OVERRIDES}; a worktree would be checked against the live tree"
    )
    existed = TOOL_OVERRIDES_PATH.exists()
    before = TOOL_OVERRIDES_PATH.read_bytes() if existed else None
    try:
        save_tool_overrides()
        assert TOOL_OVERRIDES_PATH.exists(), "the writer produced no file"
        assert not _git("status", "--porcelain", OVERRIDES).stdout.strip(), (
            "writing the override file must not register in `git status` at all"
        )

        # The atomic writer lands a sibling `.<pid>.tmp` and renames it. A
        # write killed in between leaves that behind, and an unignored stray
        # dirties the tree exactly as the tracked file used to — same outage,
        # one filename over.
        stray = TOOL_OVERRIDES_PATH.with_name(
            f"{TOOL_OVERRIDES_PATH.name}.{os.getpid()}.tmp")
        stray.write_text("", encoding="utf-8")
        try:
            assert not _git("status", "--porcelain",
                            str(stray.relative_to(ROOT))).stdout.strip(), (
                "a half-written override temp file dirties the tree"
            )
        finally:
            stray.unlink()
    finally:
        if before is None:
            TOOL_OVERRIDES_PATH.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                TOOL_OVERRIDES_PATH.parent.rmdir()
        else:
            TOOL_OVERRIDES_PATH.write_bytes(before)


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
