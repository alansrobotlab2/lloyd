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

#2756 turned the walk around. A unit that lives in the deploy directory with no
same-named tracked file behind it is now graded too, as `orphanunit:<name>`:
`untracked:` and unhealthy — nothing in the repo owns the unit that runs — unless the
name carries a disposition in the installer's `HOST_LOCAL_UNITS` array, which the row
reads ok and quotes the one-line why written as that array's comment. The node that
pinned the old one-way blind spot, `test_a_deployed_unit_the_tree_does_not_track_gets_no_row`,
is replaced by `test_a_deployed_unit_the_tree_does_not_track_gets_exactly_one_row_each`;
its "deliberately out of scope" reading of #2714 no longer holds.
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

    def host_local(self, name: str, why: str) -> None:
        """Record a disposition the way the real installer records one (#2756):
        `HOST_LOCAL_UNITS=(...)` with one deployed unit name per line and the one-line
        why as that line's `#` comment, read back by `_host_local_dispositions`.
        """
        self.installer.write_text(
            INSTALLER_STUB + f"HOST_LOCAL_UNITS=(\n    {name}  # {why}\n)\n",
            encoding="utf-8")

    def deployed_subdir(self, name: str) -> Path:
        """A directory inside the deploy directory. systemd itself maintains these —
        `timers.target.wants/` holds the armed symlinks, `<unit>.d/` the drop-ins —
        and none of them is a unit file the reverse walk may read.
        """
        path = self.deployed / name
        path.mkdir(parents=True)
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


def test_a_deployed_unit_the_tree_does_not_track_gets_exactly_one_row_each(fleet):
    """#2756 clauses 1 and 4, replacing #2714's pinned blind spot: the walk now goes
    deploy→repo as well, and every planted unit that lives in the deploy directory
    with no same-named tracked file produces exactly one row, named `orphanunit:<name>`,
    whose status names the deployed path and states it is tracked nowhere in the repo.

    The three files this box really has are the planted ones: two plain copies
    (`lloyd-graph-backup.{service,timer}`) and a dangling symlink
    (`freeclip-a2dp-fix.service`) — a dangling link is still a live directory entry,
    so it is graded too and must not raise. No disposition is recorded here, so all
    three rows are unhealthy; the disposition half of the rule is its own node, and
    the duplicate-proof half against systemd's own subdirectories is the node after
    this one.
    """
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")
    fleet.copy("lloyd-graph-backup.service")
    fleet.copy("lloyd-graph-backup.timer", TIMER_BODY)
    fleet.link("freeclip-a2dp-fix.service", fleet.tree / "nowhere.service")

    rows = fleet.rows()
    assert len(rows) == len({r["name"] for r in rows}), (
        "a deployed unit produced more than one row: "
        f"{sorted(r['name'] for r in rows)}")
    assert fleet.names() == {
        "unitlink:alpha.service",
        "orphanunit:lloyd-graph-backup.service",
        "orphanunit:lloyd-graph-backup.timer",
        "orphanunit:freeclip-a2dp-fix.service"}

    for name in ("lloyd-graph-backup.service", "lloyd-graph-backup.timer",
                 "freeclip-a2dp-fix.service"):
        row = next(r for r in rows if r["name"] == f"orphanunit:{name}")
        assert row["healthy"] is False and row["exit_code"] == 1, row
        assert row["category"] == shc.DEPLOY_CATEGORY, row
        assert str(fleet.deployed / name) in row["status"], row
        assert "tracked nowhere in the repo" in row["status"], row


def test_the_reverse_walk_reads_unit_files_one_level_deep_not_directories(fleet):
    """#2756 clause 2: the armed-timer symlinks under `timers.target.wants/` and a
    `*.d` drop-in directory produce no extra and no duplicate row, and a tracked
    unit's own deployed symlink is counted once, by the forward walk.

    This box keeps seven symlinks in `timers.target.wants/`; the planted one points
    at the top-level unit also planted here, so it is the strongest duplicate the
    deployed side can offer. `alpha.service.d` is the drop-in beside the unit it
    extends, and `stray.service` is a DIRECTORY named like a unit — the one thing a
    bare name-glob would hand back as a row. The expected set is what proves all
    three exclusions at once: any leak shows up as a name or a duplicate.
    """
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")
    fleet.copy("lloyd-graph-backup.timer", TIMER_BODY)
    wants = fleet.deployed_subdir("timers.target.wants")
    (wants / "lloyd-graph-backup.timer").symlink_to(
        fleet.deployed / "lloyd-graph-backup.timer")
    dropin = fleet.deployed_subdir("alpha.service.d")
    (dropin / "override.conf").write_text("[Service]\n", encoding="utf-8")
    fleet.deployed_subdir("stray.service")

    assert fleet.names() == {
        "unitlink:alpha.service",
        "orphanunit:lloyd-graph-backup.timer"}, fleet.names()


def test_an_untracked_deployed_unit_appearing_after_the_first_call_is_still_graded(fleet):
    """#2756 clause 2's call-time half: the deployed set is globbed when the check is
    called, so a unit dropped into the directory between calls is graded on the next
    one with nobody editing the checker — the same property the first node pins for
    the tree side.
    """
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")
    assert "orphanunit:dropped-in-later.timer" not in fleet.names()

    fleet.copy("dropped-in-later.timer", TIMER_BODY)
    assert "orphanunit:dropped-in-later.timer" in fleet.names(), (
        "a unit dropped into the deploy directory after the first call was not "
        "graded, so the deployed set is not globbed at call time")


def test_an_orphan_with_a_recorded_disposition_reads_ok_and_quotes_its_why(fleet):
    """#2756 clause 3: an orphan named in the installer's `HOST_LOCAL_UNITS` is a
    deliberate gap rather than drift — the row reads ok and quotes the one-line why
    carried as the array's comment, read from the installer's source the same way
    `SYSTEM_SCOPE_UNITS` is. An orphan the list does not name still reads unhealthy.

    The disposition and its absence are asserted in one node because the whole rule
    is the difference between the two rows: a list that made every orphan ok, or one
    whose comment the row never quotes, fails here.
    """
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")
    fleet.copy("freeclip-a2dp-fix.service")
    fleet.copy("lloyd-graph-backup.service")
    fleet.host_local("freeclip-a2dp-fix.service",
                     "host workaround for the FreeClip-2 buds; no Lloyd script owns it")

    rows = {r["name"]: r for r in fleet.rows()}
    assert set(rows) == {"unitlink:alpha.service",
                         "orphanunit:freeclip-a2dp-fix.service",
                         "orphanunit:lloyd-graph-backup.service"}
    ok = rows["orphanunit:freeclip-a2dp-fix.service"]
    assert ok["healthy"] is True and ok["exit_code"] == 0, ok
    assert ok["status"].startswith("ok:"), ok
    assert str(fleet.deployed / "freeclip-a2dp-fix.service") in ok["status"], ok
    assert "host workaround for the FreeClip-2 buds" in ok["status"], ok
    fault = rows["orphanunit:lloyd-graph-backup.service"]
    assert fault["healthy"] is False and fault["exit_code"] == 1, fault


def test_an_undispositioned_orphan_turns_the_deploy_heading_degraded_through_main(
        fleet, monkeypatch, capsys):
    """#2756 clause 3's consequence, through `main()`: the undispositioned orphan is a
    graded member of the `deploy` category, so the heading word the reader gets is
    `degraded` while no disposition names it, and `healthy` once every orphan in the
    directory carries one.

    The rows are the real function's rows over the synthetic fleet — the patch only
    re-points its three path arguments, and the real function is captured BEFORE the
    patch because a lambda that reached the module attribute would call itself — and
    `check_deployed_copies` is silenced because its byte pairs read `/etc`, which is
    not this heading's question here. Same drive shape
    `tests/test_service_health_check_path_sandbox.py::_drive` uses: argv in, stdout
    out, the payload's own `summary` read back.
    """
    fleet.track("alpha.service")
    fleet.link_to_tree("alpha.service")
    fleet.copy("lloyd-graph-backup.timer", TIMER_BODY)

    real = shc.check_unit_file_links

    def _deploy_payload():
        monkeypatch.setattr(shc, "check_unit_file_links",
                            lambda: real(tree_dir=fleet.tree,
                                         deployed_dir=fleet.deployed,
                                         installer=fleet.installer))
        monkeypatch.setattr(shc, "check_deployed_copies", lambda: [])
        monkeypatch.setattr(shc, "_switched_off", lambda: set())
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--category",
                                          shc.DEPLOY_CATEGORY, "--format", "json"])
        shc.main()
        return json.loads(capsys.readouterr().out)

    doc = _deploy_payload()
    assert sorted(s["name"] for s in doc["services"]) == [
        "orphanunit:lloyd-graph-backup.timer",
        "unitlink:alpha.service"], doc["services"]
    assert doc["summary"][shc.DEPLOY_CATEGORY] == "degraded", doc["summary"]

    fleet.host_local("lloyd-graph-backup.timer",
                     "owed a tracking ruling (#2756); live on one box only")
    doc = _deploy_payload()
    assert doc["summary"][shc.DEPLOY_CATEGORY] == "healthy", doc["summary"]


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
    tracked user-scope unit file in the tree, the exclusions removed, plus one row per
    unit file living in the deploy directory that the tree does not track (#2756) —
    and the row shape, which is what keeps the two formatters and the category summary
    free of a special case.
    """
    assert shc.SYSTEMD_UNIT_TREE == ROOT / "agent-services" / "systemd"
    assert shc.USER_UNIT_DIR == Path.home() / ".config" / "systemd" / "user"

    rows = shc.check_unit_file_links()
    assert rows, "the default run graded no unit at all"
    tracked = {p.name
               for pattern in ("*.service", "*.timer")
               for p in shc.SYSTEMD_UNIT_TREE.glob(pattern)}
    want = {f"unitlink:{name}" for name in tracked}
    want -= {f"unitlink:{name}" for name in shc._system_scope_units()}
    want |= {f"orphanunit:{p.name}"
             for pattern in ("*.service", "*.timer")
             for p in shc.USER_UNIT_DIR.glob(pattern)
             if not p.is_dir() and p.name not in tracked}
    assert {r["name"] for r in rows} == want, (
        "the graded set drifted away from the two directories it is supposed to glob")

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
    shares a category heading with. Whether THIS box currently has an untracked unit is
    host state, so #2756's `orphanunit:` rows are shape-checked where they appear and
    never presence-asserted here.
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
    for row in payload["services"]:
        if not row["name"].startswith("orphanunit:"):
            continue
        assert row["category"] == shc.DEPLOY_CATEGORY, row
        assert row["status"].split(":")[0] in {"ok", "untracked"}, row
        assert row["healthy"] is row["status"].startswith("ok:"), row
        assert "tracked nowhere in the repo" in row["status"], row
