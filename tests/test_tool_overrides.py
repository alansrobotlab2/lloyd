"""`data/tool_overrides.yaml` — the UI-mutable slice, and why it is not source.

Two properties, and they are the same property seen from two sides.

The Tools page must be able to change tool state without editing
`config.yaml`, because `config.yaml` is tracked and a tracked file rewritten
by a UI click leaves the live tree dirty — which `scripts/automod/gate.py`
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
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

from app.config import _merge_tool_overrides  # noqa: E402
from scripts.maintenance.sync_tool_overrides import (  # noqa: E402
    overrides_path, parity, sync)

OVERRIDES = "data/tool_overrides.yaml"
SERVER = "lloyd-mcp"


def _warnings(caplog) -> list[str]:
    """The warning lines this module's logger actually emitted."""
    return [r.getMessage() for r in caplog.records if r.name == "lloyd-config"]


def _pair(tmp_path, *, tracked: dict, served: dict) -> Path:
    """A checkout tree at `tmp_path` carrying `tracked` as config.yaml.

    The override goes where that tree's override would be served from
    (`overrides_path`), which for a scratch tree is its own `.lloyd-data/`. The
    point of building both files rather than passing dicts to `parity` is that
    the comparison then runs over the real document shape — the same pair of
    YAML files a box has, at a path a test may lawfully write.
    """
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.dump(tracked, sort_keys=False))
    ovr = overrides_path(root)
    ovr.parent.mkdir(parents=True, exist_ok=True)
    ovr.write_text(yaml.dump(served, sort_keys=False))
    return root


def _read_pair(root: Path) -> tuple[dict, dict]:
    return (yaml.safe_load((root / "config.yaml").read_text()) or {},
            yaml.safe_load(overrides_path(root).read_text()) or {})


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
        f"and the automod gate/promoter refuse to run."
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
    module exists to prevent. Since 2026-09-22 the file lives under the data
    root, outside the tree (`architecture/data-home.md`); the check is the same
    one — the whole tree's `git status` does not move — and holds either way.
    """
    from app.config import TOOL_OVERRIDES_PATH, save_tool_overrides
    from app.paths import DATA_ROOT

    assert TOOL_OVERRIDES_PATH == DATA_ROOT / OVERRIDES, (
        f"the writer targets {TOOL_OVERRIDES_PATH}, not the data root's {DATA_ROOT / OVERRIDES}")
    status_before = _git("status", "--porcelain").stdout
    existed = TOOL_OVERRIDES_PATH.exists()
    before = TOOL_OVERRIDES_PATH.read_bytes() if existed else None
    try:
        save_tool_overrides()
        assert TOOL_OVERRIDES_PATH.exists(), "the writer produced no file"
        assert _git("status", "--porcelain").stdout == status_before, (
            "writing the override file must not register in `git status` at all"
        )

        # The atomic writer lands a sibling `.<pid>.tmp` and renames it. A
        # write killed in between leaves that behind; it must not dirty the tree.
        stray = TOOL_OVERRIDES_PATH.with_name(
            f"{TOOL_OVERRIDES_PATH.name}.{os.getpid()}.tmp")
        stray.write_text("", encoding="utf-8")
        try:
            assert _git("status", "--porcelain").stdout == status_before, (
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
# The same disagreement at server level — in both directions
# ---------------------------------------------------------------------------


def _server_config(*, disabled: list[str], enabled: bool | None = None) -> dict:
    """A boot config with one MCP server, the shape `config.yaml` has.

    `enabled` is only spelled when asked for, because config.yaml does not spell
    it for `lloyd-mcp` today — and an override that states a key the tracked file
    leaves unstated is not a disagreement (the node below pins that).
    """
    server: dict = {"name": "lloyd-mcp", "type": "http",
                    "disabled_tools": list(disabled)}
    if enabled is not None:
        server["enabled"] = enabled
    return {"mcp_servers": {SERVER: server}, "harness": {"tool_search": {}}}


def _write_override(tmp_path, body: str) -> Path:
    ovr = tmp_path / "tool_overrides.yaml"
    ovr.write_text(body, encoding="utf-8")
    return ovr


def test_a_disable_the_override_adds_is_logged_by_name(tmp_path, monkeypatch, caplog):
    """§8's invariant from the side this box actually lives on.

    `architecture/tools.md` §8 records that the override on this machine disables
    fifteen mail mutators config.yaml never lists. The merge computed only
    `was - now`, so a tool could become disabled with no tracked record of it:
    the tracked file went on implying a wider served surface than the box offers,
    and a rebuild boots the mutators the operator switched off (#3128). The
    override still wins — the warning reports the disagreement, it does not veto
    it — and it has to name which tools, because "some tools" is not a thing
    anyone can go reconcile.
    """
    ovr = _write_override(
        tmp_path, "mcp_servers:\n  lloyd-mcp:\n    disabled_tools: [alpha, email_delete]\n")
    monkeypatch.setattr("app.config.TOOL_OVERRIDES_PATH", ovr)

    with caplog.at_level(logging.WARNING, logger="lloyd-config"):
        merged = _merge_tool_overrides(_server_config(disabled=["alpha"]))

    assert merged["mcp_servers"][SERVER]["disabled_tools"] == ["alpha", "email_delete"], (
        "the override must still win: the Tools page is the live authority")
    text = "\n".join(_warnings(caplog))
    assert "email_delete" in text, f"the disagreeing tool must be named; got {text!r}"
    assert "alpha" not in text, "the tools the two files agree on are not drift"
    assert "re-enables" not in text, "nothing was resurrected in this pair"


def test_a_tool_the_override_resurrects_is_still_logged(tmp_path, monkeypatch, caplog):
    """The direction warned since 2026-09-04, pinned so the new one cannot
    displace it: `browser_screenshot` sat in config.yaml's list while the
    override kept it advertised."""
    ovr = _write_override(
        tmp_path, "mcp_servers:\n  lloyd-mcp:\n    disabled_tools: [alpha]\n")
    monkeypatch.setattr("app.config.TOOL_OVERRIDES_PATH", ovr)

    with caplog.at_level(logging.WARNING, logger="lloyd-config"):
        merged = _merge_tool_overrides(
            _server_config(disabled=["alpha", "browser_screenshot"]))

    assert merged["mcp_servers"][SERVER]["disabled_tools"] == ["alpha"]
    text = "\n".join(_warnings(caplog))
    assert "browser_screenshot" in text, f"got {text!r}"
    assert "disables" not in text, "this pair adds no disable, only removes one"


def test_a_per_server_enabled_disagreement_is_logged(tmp_path, monkeypatch, caplog):
    """One file says the server is switched off; the box serves it.

    The merge assigned `cfg["enabled"]` with no comparison at all, so this — the
    widest disagreement the block can hold, wider than any single tool — was the
    one class that could not be seen from the boot log.
    """
    ovr = _write_override(tmp_path, f"mcp_servers:\n  {SERVER}:\n    enabled: true\n")
    monkeypatch.setattr("app.config.TOOL_OVERRIDES_PATH", ovr)

    with caplog.at_level(logging.WARNING, logger="lloyd-config"):
        merged = _merge_tool_overrides(_server_config(disabled=[], enabled=False))

    assert merged["mcp_servers"][SERVER]["enabled"] is True, "the override still wins"
    text = "\n".join(_warnings(caplog))
    assert "enabled" in text and SERVER in text, (
        f"the warning must name the server and the key; got {text!r}")


def test_an_agreeing_server_pair_logs_nothing(tmp_path, monkeypatch, caplog):
    """The steady state stays silent, in both directions at once.

    The Tools page writes the whole server block back on every toggle, so an
    override that restates config.yaml exactly is the normal case. Warning on it
    would fire on every boot and stop meaning anything — which is the same reason
    `test_agreeing_files_log_nothing` exists for `tool_search`.
    """
    ovr = _write_override(tmp_path,
                    f"mcp_servers:\n  {SERVER}:\n    enabled: true\n"
                    "    disabled_tools: [alpha, beta]\n")
    monkeypatch.setattr("app.config.TOOL_OVERRIDES_PATH", ovr)

    with caplog.at_level(logging.WARNING, logger="lloyd-config"):
        merged = _merge_tool_overrides(
            _server_config(disabled=["alpha", "beta"], enabled=True))

    assert merged["mcp_servers"][SERVER]["disabled_tools"] == ["alpha", "beta"]
    assert not _warnings(caplog), f"no disagreement, so nothing to report; got {_warnings(caplog)!r}"


def test_an_enabled_key_config_never_states_is_not_a_disagreement(tmp_path, monkeypatch, caplog):
    """config.yaml spells no `enabled` for `lloyd-mcp`, which is not the same as
    saying `true`: an override that states an unstated key is adding a default,
    and the rule `test_a_key_absent_from_config_is_not_a_disagreement` pins for
    `tool_search` holds here too — otherwise the warning fires on a box whose
    tracked file simply never took a position.
    """
    ovr = _write_override(tmp_path, f"mcp_servers:\n  {SERVER}:\n    enabled: false\n")
    monkeypatch.setattr("app.config.TOOL_OVERRIDES_PATH", ovr)

    with caplog.at_level(logging.WARNING, logger="lloyd-config"):
        merged = _merge_tool_overrides(_server_config(disabled=[]))

    assert merged["mcp_servers"][SERVER]["enabled"] is False
    assert not _warnings(caplog), f"got {_warnings(caplog)!r}"


# ---------------------------------------------------------------------------
# The tracked defaults must describe what a rebuild would actually serve
# ---------------------------------------------------------------------------


def test_config_yaml_agrees_with_the_live_override(tmp_path):
    """The comparison that answers that question, pinned on a pair it can read.

    It used to open `ROOT / "data/tool_overrides.yaml"` and `return` when the file
    was absent. Runtime data left the code tree in `6426668b` (2026-09-22), so that
    path has existed in no checkout since — a round's worktree AND the live box —
    which means the early return fired everywhere and §8's "tracked describes
    served" invariant had a guard that asserted nothing while still printing PASSED.
    The suite cannot read the box's own file by any route either: `tests/conftest.py`
    repoints `LLOYD_DATA` at a scratch data root and refuses to run against the
    production tree.

    So this node pins the instrument — `sync_tool_overrides.parity`, the one
    function the command runs — on an injected pair, and the box itself is asked
    outside pytest by that command. For the command's answer to mean anything, the
    comparison has to be able to fail: one extra disable in the override is drift,
    a dropped one is drift, and an override that restates config.yaml is not.

    SETUP.md tells you to restore the override from backup and warns that without
    it tool state falls back to config.yaml's defaults, so a disagreement is not
    confusing documentation — it is the state a fresh clone boots into.
    """
    added = _pair(tmp_path / "added",
                  tracked={"mcp_servers": {SERVER: {"disabled_tools": ["alpha"]}}},
                  served={"mcp_servers": {SERVER:
                                          {"disabled_tools": ["alpha", "email_delete"]}}})
    report = parity(*_read_pair(added))
    key = f"mcp_servers.{SERVER}.disabled_tools"
    assert list(report) == [key], f"the extra disable must be reported; got {report!r}"
    assert report[key] == (["alpha"], ["alpha", "email_delete"]), (
        f"(tracked, served) for the disagreeing set; got {report[key]!r}")

    dropped = _pair(tmp_path / "dropped",
                    tracked={"mcp_servers": {SERVER:
                                             {"disabled_tools": ["alpha", "beta"]}}},
                    served={"mcp_servers": {SERVER: {"disabled_tools": ["alpha"]}}})
    assert parity(*_read_pair(dropped)) == {key: (["alpha", "beta"], ["alpha"])}, (
        "a disable config.yaml still lists but the override dropped is drift too")

    identical = _pair(tmp_path / "identical",
                      tracked={"mcp_servers": {SERVER: {"disabled_tools": ["alpha"]}}},
                      served={"mcp_servers": {SERVER: {"disabled_tools": ["alpha"]}}})
    assert parity(*_read_pair(identical)) == {}, (
        "an override restating the tracked set is the steady state, not drift")


# ---------------------------------------------------------------------------
# The probe is the only thing that can ask about this box, so it must be able
# to answer — and to say so in an exit code
# ---------------------------------------------------------------------------


def _mail_mutator_pair(tmp_path, *, extra: bool = True) -> Path:
    """A tree whose override disables `email_delete` on top of config.yaml's set.

    `extra=False` leaves the two files identical, which is the half that has to
    stay green or a widened report is just a different false alarm.
    """
    tracked_tools = ["discord_send"]
    return _pair(
        tmp_path,
        tracked={"mcp_servers": {SERVER: {"disabled_tools": tracked_tools}}},
        served={"mcp_servers": {SERVER: {"disabled_tools":
                                        tracked_tools + (["email_delete"] if extra else [])}}})


def test_the_probe_names_a_disable_the_tracked_config_omits_and_exits_1(tmp_path, capsys):
    """The input that used to print `already in sync` and exit 0.

    The command compared `harness.tool_search` alone, so a box whose override
    disables fifteen tools config.yaml does not list was "in sync" — and its own
    docstring offered `--check` as a pre-merge probe, which is a gate wired to an
    instrument that cannot fail. Exit code is the answer, and the tool name has to
    be on stdout for the answer to be actionable.
    """
    root = _mail_mutator_pair(tmp_path)
    before = overrides_path(root).read_text()

    assert sync(root, check=True) == 1, "--check must exit non-zero on drift"
    out = capsys.readouterr().out
    assert "email_delete" in out, f"the divergent tool must be named; got {out!r}"
    assert overrides_path(root).read_text() == before, "--check must not write"


def test_the_probe_exits_zero_on_an_identical_pair(tmp_path, capsys):
    """Silence has to stay available: the same command that reports the fifteen has
    to report nothing on a box whose config.yaml lists what it serves, or #3382's
    commit could never make it green."""
    root = _mail_mutator_pair(tmp_path, extra=False)
    assert sync(root, check=True) == 0
    assert "already in sync" in capsys.readouterr().out


def test_the_probe_names_a_resurrected_tool_too(tmp_path, capsys):
    """Both directions, in one report: a tracked file listing a disable the
    override dropped is the same class of lie pointing the other way, and `+` is
    how it says so."""
    root = _pair(
        tmp_path,
        tracked={"mcp_servers": {SERVER:
                                 {"disabled_tools": ["discord_send", "browser_screenshot"]}}},
        served={"mcp_servers": {SERVER: {"disabled_tools": ["discord_send"]}}})
    assert sync(root, check=True) == 1
    out = capsys.readouterr().out
    assert "+ browser_screenshot" in out, (
        f"tracked-but-not-served prints as a `+`; got {out!r}")


def test_the_probe_refuses_to_rewrite_a_tool_disable(tmp_path, capsys):
    """Without `--check` it repairs `harness.tool_search` only. For a disable-set
    the override wins by design, so copying config.yaml's narrower set into the
    override would re-enable tools the operator switched off — the command prints
    the disagreement, says config.yaml is the file to edit, and stays non-zero.
    """
    root = _mail_mutator_pair(tmp_path)
    before = overrides_path(root).read_text()

    assert sync(root) == 1
    assert overrides_path(root).read_text() == before, "the disables must not be rewritten"
    assert "email_delete" in capsys.readouterr().out


def test_a_server_only_the_override_names_is_reported(tmp_path, capsys):
    """The class the merge hides by construction: `app/config.py` skips an override
    server config.yaml has no block for, so nothing is served from it and nothing
    is logged about it. Usually a renamed server — worth a line, not the silence.
    """
    root = _pair(tmp_path,
                 tracked={"mcp_servers": {"other-mcp": {"disabled_tools": []}}},
                 served={"mcp_servers": {SERVER: {"disabled_tools": []}}})
    assert sync(root, check=True) == 1
    out = capsys.readouterr().out
    assert SERVER in out, f"the server config.yaml has no block for must be named; got {out!r}"


def test_the_cli_itself_exits_1_on_the_divergent_pair(tmp_path):
    """The invocation a human or a gate actually runs, across the process boundary.

    `sync()` returning 1 is not the same fact as `python -m …` exiting 1: the
    wiring in between is `argparse` plus `sys.exit(main())`, and the live-box
    check the owed note asks for is that command line, not the function. So this
    runs the module as a child against a fixture tree and reads the exit status
    and the child's own stdout.
    """
    root = _mail_mutator_pair(tmp_path)
    r = subprocess.run(
        [sys.executable, "-m", "scripts.maintenance.sync_tool_overrides",
         "--check", str(root)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 1, f"stdout={r.stdout!r} stderr={r.stderr[-500:]!r}"
    assert "email_delete" in r.stdout, f"stdout={r.stdout!r}"


def test_a_workers_enabled_disagreement_is_reported(tmp_path, capsys):
    """`workers.enabled` is the pool's master switch, and the override is its
    authority too, so the probe reports it on the same terms."""
    root = _pair(tmp_path,
                 tracked={"workers": {"enabled": True}},
                 served={"workers": {"enabled": False}})
    assert sync(root, check=True) == 1
    assert "workers.enabled" in capsys.readouterr().out
