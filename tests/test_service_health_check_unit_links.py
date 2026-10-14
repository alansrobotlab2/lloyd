"""#2714: every tracked user unit file is graded for being the file it claims to be.

`agent-services/setup/install-services.sh:55-75` symlinks every `.service` and `.timer`
in `agent-services/systemd/` into `~/.config/systemd/user/`, and that symlink is the
whole reason `architecture/infrastructure.md` may say "editing the repo copy is the
deploy". On this box five tracked units are not symlinks. They are plain `-rw-r--r--`
copies placed by hand before the installer last ran: `lloyd-cert-renew.{service,timer}`,
`lloyd-egress-window.{service,timer}`, `lloyd-persistence-window.{service,timer}`,
`thunderbird.service` and `voxtype.service` — eight deployed files, all eight tracked in
the tree. Their bytes match the tree today, so `check_deployed_copies`, which sha256-
compares the one hand-installed root pair, reports `Deploy: healthy` while a repo edit
to any of those eight files lands nowhere. **Identity** is the property, not content,
and nothing checked it.

`check_unit_file_links` is that check. Each node below names the acceptance clause it
pins. Every node runs over a synthetic unit directory and a synthetic deploy directory
under `tmp_path`, so no node reads this box's unit state and no node needs the installer
to have been run — which is also why the units have invented names: clause 1 is that a
unit added to the directory is graded with nobody editing the checker, and a test built
on this box's own eighteen user-scope files could not show that. The one node that does
touch the real tree asserts only what stays true whichever way the deploy directory
falls, because the owed installer re-run is going to move those verdicts.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"


def _load_module():
    """Load the CLI script by path: it has no package and nothing imports it."""
    spec = importlib.util.spec_from_file_location("shc_unit_links", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shc = _load_module()

#: A stand-in installer carrying the exclusion line the real one carries at
#: `install-services.sh:49`, in the same shape, so the parse is graded on the shape and
#: not on the real file's length. `test_the_exclusion_list_is_the_installer_s` re-reads
#: the real script beside it.
INSTALLER_STUB = """#!/usr/bin/env bash
SYSTEMD_SRC="$PROJECT_DIR/systemd"
SYSTEMD_DST="$HOME/.config/systemd/user"
SYSTEM_SCOPE_UNITS=(nvidia-power-limit.service)

for f in "$SYSTEMD_SRC"/*.service "$SYSTEMD_SRC"/*.timer; do
    continue
done
"""

UNIT_BODY = "[Unit]\nDescription=synthetic\n\n[Service]\nType=oneshot\nExecStart=/bin/true\n"
TIMER_BODY = ("[Unit]\nDescription=synthetic timer\n\n"
              "[Timer]\nOnCalendar=*-*-* 04:00:00\n")


class _Fleet:
    """A synthetic repo unit directory and the deploy directory it links into.

    The two directories mirror the real layout (`agent-services/systemd/` and
    `~/.config/systemd/user/`) rather than being two flat temp dirs, so a relative
    symlink — which is a thing `ln` can produce and the check has to resolve — has a
    parent chain to resolve against.
    """

    def __init__(self, root: Path):
        self.tree = root / "repo" / "agent-services" / "systemd"
        self.tree.mkdir(parents=True)
        self.deployed = root / "user" / ".config" / "systemd" / "user"
        self.deployed.mkdir(parents=True)
        self.installer = root / "install-services.sh"
        self.installer.write_text(INSTALLER_STUB, encoding="utf-8")

    def track(self, name: str, body: str = None) -> Path:
        """A tracked unit file in the tree, at `agent-services/systemd/<name>`."""
        path = self.tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body or UNIT_BODY, encoding="utf-8")
        return path

    def link(self, name: str, target: Path) -> Path:
        """The deployed path as a symlink to `target`, as the installer leaves it."""
        path = self.deployed / name
        path.symlink_to(target)
        return path

    def link_to_tree(self, name: str) -> Path:
        """The healthy state: a symlink at the deployed path pointing at the tracked file."""
        return self.link(name, self.tree / name)

    def link_relative(self, name: str) -> Path:
        """The same link written relative to the deploy directory, not absolute."""
        return self.link(name, Path(os.path.relpath(self.tree / name, self.deployed)))

    def copy(self, name: str, body: str = None) -> Path:
        """The deployed path as a plain file — the #2714 state, byte-identical by default."""
        path = self.deployed / name
        path.write_text(body or (TIMER_BODY if name.endswith(".timer") else UNIT_BODY),
                        encoding="utf-8")
        return path

    def rows(self) -> list:
        return shc.check_unit_file_links(tree_dir=self.tree,
                                        deployed_dir=self.deployed,
                                        installer=self.installer)

    def names(self) -> set:
        return {r["name"] for r in self.rows()}


@pytest.fixture
def fleet(tmp_path: Path) -> _Fleet:
    return _Fleet(tmp_path)


def _one(rows: list) -> dict:
    assert len(rows) == 1, rows
    return rows[0]


# ---------------------------------------------------------------------------
# Clause 1 — the graded set is the directory, read at call time
# ---------------------------------------------------------------------------

def test_a_unit_this_box_has_never_had_is_graded_with_no_edit_to_the_checker(fleet):
    """Clause 1: the units graded come from a glob over `*.service` and `*.timer`
    taken when the check runs, so a new tracked unit is graded on the next call.

    `brand-new-thing.timer` is a unit this machine has never had and never will: the
    checker has no name to have been written with, so a row for it can only have come
    from the directory. The second half is the same property one call later — a unit
    that appears in the tree AFTER the checker exists still gets graded, which is what
    rules out a list someone has to append to.
    """
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")
    fleet.track("brand-new-thing.timer", TIMER_BODY)
    fleet.copy("brand-new-thing.timer", TIMER_BODY)

    assert fleet.names() == {"unitlink:alpha.service",
                             "unitlink:brand-new-thing.timer"}

    fleet.track("added-afterwards.service")
    assert "unitlink:added-afterwards.service" in fleet.names(), (
        "a unit added to the directory was not graded, so the graded set is not the "
        "directory")


def test_a_deployed_unit_the_tree_does_not_track_gets_no_row(fleet):
    """Clause 1's direction: the check walks repo to deploy, never the reverse.

    `lloyd-graph-backup.{service,timer}` and `freeclip-a2dp-fix.service` live on this
    box and are tracked nowhere, which #2714 puts deliberately out of scope — the same
    membership-not-equality ruling `tests/test_data_home.py:1355` already makes about
    that timer. An untracked plain copy and a dangling one are both planted here: the
    tree carries one tracked unit, and that unit is the only row.
    """
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")
    fleet.copy("lloyd-graph-backup.service")
    fleet.copy("lloyd-graph-backup.timer", TIMER_BODY)
    fleet.link("freeclip-a2dp-fix.service", fleet.tree / "nowhere.service")

    assert fleet.names() == {"unitlink:alpha.service"}


# ---------------------------------------------------------------------------
# Clause 2 — a plain copy is the fault, and a byte-identical one
# ---------------------------------------------------------------------------

def test_a_plain_copy_with_identical_bytes_is_reported_naming_both_paths(fleet):
    """Clause 2: a regular file where the installer would place a symlink is
    unhealthy, and the line names the deployed path, the tracked path, and the fact
    that it is not a link to the tree.

    The bytes are deliberately IDENTICAL — `tracked.read_bytes()` is compared against
    the copy below, and the status is asserted to say nothing about drift — because
    that is exactly the state of all eight deployed copies on this box today. A row
    phrased as a content difference would send the reader to reinstall content that
    already matches and would let them conclude the alarm was a false one.
    """
    tracked = fleet.track("lloyd-cert-renew.service")
    deployed = fleet.copy("lloyd-cert-renew.service")
    assert tracked.read_bytes() == deployed.read_bytes(), (
        "the fixture stopped being the state the item describes")

    row = _one(fleet.rows())
    assert row["healthy"] is False, row
    assert row["exit_code"] == 1, row
    assert row["category"] == shc.DEPLOY_CATEGORY
    assert str(deployed) in row["status"] and str(tracked) in row["status"], row
    assert "not a link to the tree" in row["status"], row
    assert "drift" not in row["status"], (
        "the copy's bytes match the tree; calling this drift sends the reader to "
        f"reinstall content that already matches: {row['status']}")
    assert "install-services.sh" in row["status"], (
        "the line has to name the thing that would fix it: " + row["status"])


def test_a_plain_copy_is_reported_even_though_the_tree_unit_is_a_service_not_a_timer(fleet):
    """Clause 2 over both globs: one faulting `.service` and one faulting `.timer`
    produce two rows, so neither glob was bolted on after the first."""
    fleet.track("synth-a.service")
    fleet.copy("synth-a.service")
    fleet.track("synth-b.timer", TIMER_BODY)
    fleet.copy("synth-b.timer", TIMER_BODY)

    rows = {r["name"]: r for r in fleet.rows()}
    assert set(rows) == {"unitlink:synth-a.service", "unitlink:synth-b.timer"}
    assert all(r["healthy"] is False for r in rows.values()), rows


def test_a_tracked_unit_with_nothing_deployed_at_all_is_reported(fleet):
    """The third way the invariant breaks: the installer never linked it, so the unit
    is not installed and never ran. `missing:` is the word `check_deployed_copies`
    already uses for a target that is not there, kept for the same reading."""
    tracked = fleet.track("never-installed.timer", TIMER_BODY)
    row = _one(fleet.rows())
    assert row["healthy"] is False, row
    assert row["status"].startswith("missing:"), row
    assert str(tracked) in row["status"], row
    assert str(fleet.deployed / "never-installed.timer") in row["status"], row


# ---------------------------------------------------------------------------
# Clause 3 — a symlink is healthy only when it points at the tree
# ---------------------------------------------------------------------------

def test_a_link_to_the_tracked_unit_is_healthy(fleet):
    """Clause 3, first half: a symlink resolving to its tracked repo file is `ok`, and
    it is `ok` whether the installer wrote it absolute (`ln -sf "$(realpath …)"`) or
    somebody made it relative by hand — the property is what it resolves to."""
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")
    fleet.track("beta.service")
    fleet.link_relative("beta.service")

    rows = {r["name"]: r for r in fleet.rows()}
    assert set(rows) == {"unitlink:alpha.service", "unitlink:beta.service"}
    for row in rows.values():
        assert row["healthy"] is True, row
        assert row["status"].startswith("ok:"), row
        assert row["exit_code"] == 0, row
        assert row["category"] == shc.DEPLOY_CATEGORY


def test_a_link_to_any_other_path_is_unhealthy_naming_that_target(fleet):
    """Clause 3, second half: a symlink that resolves elsewhere is a fault, and the
    line names the target it points at.

    The planted target carries the tracked unit's bytes exactly, so this is not a
    content difference either — it is the same file identity question, and it is the
    shape a hand-copied-then-linked or re-pointed unit takes. A dangling symlink is the
    same fault with a target that no longer exists, so it is asserted in the same node:
    the check must not fall over on it and must not read it as healthy.
    """
    tracked = fleet.track("alpha.service")
    elsewhere = fleet.tree.parent / "stale" / "alpha.service"
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    elsewhere.write_text(tracked.read_text(encoding="utf-8"), encoding="utf-8")
    fleet.link("alpha.service", elsewhere)

    fleet.track("gamma.service")
    gone = fleet.deployed.parent / "deleted-then-unlinked.service"
    fleet.link("gamma.service", gone)
    assert not gone.exists()

    rows = {r["name"]: r for r in fleet.rows()}
    assert set(rows) == {"unitlink:alpha.service", "unitlink:gamma.service"}
    assert all(r["healthy"] is False for r in rows.values()), rows

    moved = rows["unitlink:alpha.service"]
    assert not moved["status"].startswith("ok:"), moved
    assert str(elsewhere) in moved["status"], (
        f"the line must name the target the link points at: {moved['status']}")
    assert str(tracked) in moved["status"], moved
    dangling = rows["unitlink:gamma.service"]
    assert not dangling["status"].startswith("ok:"), dangling
    assert str(gone) in dangling["status"], dangling


# ---------------------------------------------------------------------------
# Clause 4 — a gap the installer leaves on purpose is never a FAIL
# ---------------------------------------------------------------------------

def test_the_units_the_installer_skips_by_scope_are_never_reported(fleet):
    """Clause 4: a name in `SYSTEM_SCOPE_UNITS` is not graded, so the one unit the
    installer `continue`s past cannot become a false FAIL for not being linked.

    `nvidia-power-limit.service` is that unit, and the copy that runs is a hand install
    under `/etc/systemd/system/` already graded by the existing byte pair. Both names
    are tracked here and deployed nowhere, which is the state that would otherwise read
    `missing:` — and `synth-root.service` is the proof that the exclusion is the
    installer's list rather than one name this file happens to know.
    """
    fleet.installer.write_text(
        INSTALLER_STUB.replace("SYSTEM_SCOPE_UNITS=(nvidia-power-limit.service)",
                               "SYSTEM_SCOPE_UNITS=(nvidia-power-limit.service "
                               "synth-root.service)"),
        encoding="utf-8")
    fleet.track("nvidia-power-limit.service")
    fleet.track("synth-root.service")
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")

    assert fleet.names() == {"unitlink:alpha.service"}, (
        "a unit the installer deliberately does not link was reported")


def test_units_under_the_system_subdirectory_are_never_reported(fleet):
    """Clause 4's second exclusion: `agent-services/systemd/system/` holds the
    root-scope units that are installed by hand (`lloyd-data-snapshot-prune.*`), and
    the installer's own glob reaches one level only.

    Both are tracked, both are deployed nowhere, and neither may be graded. They are
    named with `.service` and `.timer` suffixes so a recursive glob would have caught
    them, which is what makes this node about the glob rather than about the names.
    """
    fleet.track("system/lloyd-data-snapshot-prune.service")
    fleet.track("system/lloyd-data-snapshot-prune.timer", TIMER_BODY)
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")

    assert fleet.names() == {"unitlink:alpha.service"}, (
        "a unit under agent-services/systemd/system/ was graded")


def test_a_checker_that_cannot_read_the_exclusion_list_grades_nothing_loudly(fleet):
    """Clause 4's failure direction: with no exclusion list the check cannot tell a
    deliberate gap from a stale copy, so it files one fault row instead of grading
    every unit — including the one it must not flag.

    Two ways to lose the list: the installer is not there, and an installer with no
    `SYSTEM_SCOPE_UNITS` line in it. `missing:` and a silent empty answer are both
    wrong here; `check_unit_enabledness`'s "the guard looked at nothing" row is the
    precedent for a check that says it graded nothing.
    """
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")

    for why in ("installer absent", "installer has no exclusion line"):
        if why == "installer absent":
            fleet.installer.unlink()
        else:
            fleet.installer.write_text("#!/usr/bin/env bash\necho nothing\n",
                                       encoding="utf-8")
        row = _one(fleet.rows())
        assert row["healthy"] is False, (why, row)
        assert "unitlink:" not in row["name"], (why, row)
        assert str(fleet.installer) in row["status"], (why, row)


def test_the_exclusion_list_is_the_installer_s_and_not_a_name_written_here():
    """The derivation, graded against the real `install-services.sh:49`.

    A green suite over temp installers would still say nothing if the default read a
    file that is not the installer, exactly as #1108's sibling node says about the byte
    pairs. And the real run must not name the system-scope unit — that assertion is
    what a regression to a hardcoded set, or to no set, breaks immediately.
    """
    assert shc._system_scope_units() == {"nvidia-power-limit.service"}, (
        "the exclusion list no longer comes from install-services.sh's "
        "SYSTEM_SCOPE_UNITS line")
    assert shc.INSTALL_SERVICES == (ROOT / "agent-services" / "setup"
                                   / "install-services.sh")
    assert shc._system_scope_units(ROOT / "setup" / "does-not-exist.sh") is None

    names = {r["name"] for r in shc.check_unit_file_links()}
    assert "unitlink:nvidia-power-limit.service" not in names


# ---------------------------------------------------------------------------
# The defaults, and the row shape the formatters depend on
# ---------------------------------------------------------------------------

def test_the_default_constants_point_at_the_repo_tree_and_the_user_unit_dir():
    """The real paths the check grades by default, and the invariant over them.

    No verdict is asserted: what is linked on the machine running the suite is not the
    suite's to decide, and the owed re-run of the installer is going to move every one
    of the five plain copies. What is box-independent is the denominator — one row per
    tracked user-scope unit file in the tree, the exclusions removed and nothing else —
    and the row shape, which is what keeps the two formatters and the category summary
    free of a special case.
    """
    assert shc.SYSTEMD_UNIT_TREE == ROOT / "agent-services" / "systemd"
    assert shc.USER_UNIT_DIR == Path.home() / ".config" / "systemd" / "user"

    rows = shc.check_unit_file_links()
    assert rows, "the default run graded no unit at all"
    want = {f"unitlink:{p.name}"
            for pattern in ("*.service", "*.timer")
            for p in shc.SYSTEMD_UNIT_TREE.glob(pattern)}
    want -= {f"unitlink:{name}" for name in shc._system_scope_units()}
    assert {r["name"] for r in rows} == want, (
        "the graded set drifted away from the tree it is supposed to glob")

    for row in rows:
        assert row["category"] == shc.DEPLOY_CATEGORY, row
        assert Path(row["tracked"]).parent == shc.SYSTEMD_UNIT_TREE, row
        assert Path(row["deployed"]).parent == shc.USER_UNIT_DIR, row
        assert {"name", "status", "healthy", "exit_code", "output", "category"} <= set(row)
        assert row["healthy"] is row["status"].startswith("ok:"), row


def test_the_default_cli_run_grades_the_units_under_the_deploy_category():
    """The seam a caller actually holds: the script run as a subprocess, on its own
    defaults, in the shape `--format json` emits.

    Shape only, for the reason in the node above — but through `main()`, because a
    function that returns the right rows and is wired to nothing would pass every other
    node in this file. Clause 5's presence-and-absence rule is pinned in
    `tests/test_service_health_check_gpu_power_drift.py`, which owns the byte pairs this
    shares a category heading with.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--category", shc.DEPLOY_CATEGORY,
         "--format", "json"],
        capture_output=True, text=True, timeout=60, check=True)
    payload = json.loads(proc.stdout)
    units = [s for s in payload["services"] if s["name"].startswith("unitlink:")]
    assert units, payload["services"]
    assert {s["category"] for s in units} == {shc.DEPLOY_CATEGORY}
    for row in units:
        assert row["status"].split(":")[0] in {"ok", "missing", "copy", "wrongtarget"}, (
            row["status"])
