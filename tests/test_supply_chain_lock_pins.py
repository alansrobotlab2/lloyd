"""#1914: what the torch / setuptools pins mean for the venv the suite runs in.

`requirements.lock` moved torch 2.11.0 → 2.13.0 and setuptools 81.0.0 → 83.0.0 to clear
the last three OSV advisories. The two had to move together: torch 2.11 declared
`setuptools<82`, the only ceiling on setuptools in the whole dependency graph. The offline
scan reads the lock, not the installed tree, so a lock edit alone reads as "0 advisories"
while the venv still runs the advised versions; these nodes are about the installed side.

The lock lands before the live venv is upgraded (installing into `.venvs/lloyd` is a
hand action, SETUP.md), so the pair node accepts exactly two states and no third.
"""

from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent

#: The pair the venv ran before #1914 — the versions the advisories name.
PRE_1914 = {"torch": Version("2.11.0"), "setuptools": Version("81.0.0")}


def _lock_pins() -> dict[str, Version]:
    pins = {}
    for raw in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        name, sep, version = raw.strip().partition("==")
        if sep and not name.startswith("#"):
            pins[name.strip().lower().replace("_", "-")] = Version(version.strip())
    return pins


def _setuptools_requirements() -> list[tuple[str, Requirement]]:
    """Every installed distribution's unconditional requirement on setuptools."""
    found = []
    for dist in md.distributions():
        for raw in dist.requires or ():
            req = Requirement(raw)
            if req.name.lower() != "setuptools":
                continue
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            found.append((dist.metadata["Name"], req))
    return found


def test_no_installed_distribution_caps_setuptools_below_the_installed_version():
    """The consistency check for the one edge that made setuptools unmovable."""
    installed = Version(md.version("setuptools"))
    reqs = _setuptools_requirements()
    assert any(name.lower() == "torch" for name, _ in reqs), (
        "torch declares a setuptools requirement in every version this repo has pinned; "
        f"finding none means the scan of installed metadata is broken: {reqs}")
    broken = [f"{name} requires setuptools{req.specifier}" for name, req in reqs
              if not req.specifier.contains(installed, prereleases=True)]
    assert not broken, f"setuptools {installed} is installed, but: {broken}"


def test_the_installed_pair_is_the_locks_or_the_pre_1914_pair_never_a_mix():
    """Half an upgrade is the state to refuse: torch 2.11 beside setuptools 83 breaks
    torch's own requirement, and torch 2.13 beside setuptools 81 keeps both setuptools
    advisories while the lock claims they are gone."""
    lock = _lock_pins()
    installed = {name: Version(md.version(name)) for name in PRE_1914}
    locked = {name: lock[name] for name in PRE_1914}
    assert installed in (locked, PRE_1914), (
        f"installed {installed} is neither the lock's pair {locked} nor the pre-#1914 "
        f"pair {PRE_1914}: install the lock (SETUP.md, 'Upgrading the main venv to a "
        "moved lock')")


def test_once_the_lock_is_installed_torchs_own_cuda_pins_are_the_locks():
    """torch pins four `nvidia-*-cu13` wheels exactly; a lock that moved torch without
    re-freezing them would install a torch whose CUDA libraries the resolver replaces.
    Checked against the installed torch's metadata, so it speaks for whichever torch this
    venv runs: the lock's (then the lock must agree) or the pre-#1914 one (then the
    installed wheels must agree, and the lock is ahead of both by design)."""
    lock = _lock_pins()
    exact = {}
    for raw in md.requires("torch") or ():
        req = Requirement(raw)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        specs = list(req.specifier)
        if req.name.lower().startswith("nvidia-") and len(specs) == 1 and specs[0].operator == "==":
            exact[req.name.lower()] = Version(specs[0].version)
    assert len(exact) >= 4, f"torch's exact nvidia pins were not found: {exact}"
    for name, want in exact.items():
        assert Version(md.version(name)) == want, f"{name}: installed != torch's pin {want}"
    if Version(md.version("torch")) == lock["torch"]:
        assert {n: lock[n] for n in exact} == exact
