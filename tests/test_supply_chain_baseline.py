"""Clause 3 of #688: the committed offline advisory baseline for this tree.

This file is the only supply-chain statement the repo makes about itself, so the
two things worth pinning are that a reader can find it (tracked, not swallowed by
a gitignore rule) and that it cannot be misread as "no vulnerabilities" when no
scanner actually ran. The populated-field half — version, duration, database
freshness — is pinned in `test_supply_chain_scan.py` against a stub scanner;
pinning the *shape* here is what lets a null in the committed file be trusted as
"no instrument" rather than "broken recorder".
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import yaml

import pytest

from app.harness import supply_chain as sc


@pytest.fixture(scope="module")
def baseline() -> dict:
    path = sc.baseline_path()
    assert path.is_file(), (f"{path} must be committed: it is this tree's supply-chain "
                            "baseline. Regenerate with "
                            "`python -m app.harness.supply_chain scan --write-baseline`")
    return sc.read_baseline(path)


def test_the_baseline_lives_under_eval_supply_chain_not_pipeline(baseline):
    """`_pipeline/` is gitignored (`.gitignore:25`), so a baseline written there is
    a file that exists only on one machine and can never be reviewed."""
    path = sc.baseline_path()
    assert path.parts[-3:] == ("eval", "supply-chain", "baseline.yaml"), path
    assert "_pipeline" not in path.parts


def test_the_baseline_parses_and_declares_itself_offline(baseline):
    assert baseline["schema"] == "lloyd-supply-chain-baseline/1"
    assert baseline["offline"] is True, ("a baseline is only comparable to itself if "
                                         "it says no network was used")


def test_the_baseline_records_the_fields_a_scan_is_asked_to_report(baseline):
    for key in ("scan_status", "advisory_count", "duration_seconds"):
        assert key in baseline, key
    for key in ("name", "version", "path"):
        assert key in baseline["scanner"], key
    assert "freshness" in baseline["database"]
    assert "generated_at" in baseline


def test_the_baseline_stamp_is_offset_bearing_utc(baseline):
    """A naive timestamp in a machine-facing payload is read as UTC by every later
    reader and silently moves the clock (the 2026-09-19 `ALERT.md` lesson)."""
    stamp = dt.datetime.fromisoformat(baseline["generated_at"].replace("Z", "+00:00"))
    assert stamp.tzinfo is not None
    assert stamp.utcoffset() == dt.timedelta(0)


def test_a_scan_that_did_not_run_records_no_count_rather_than_zero(baseline):
    """The one reading this file must never support.

    `advisory_count: 0` with `scan_status: scanner_absent` would be a clean bill of
    health issued by an instrument that does not exist on this machine — the
    "guard reporting a verdict it cannot justify" class, applied to a committed
    report instead of a live check.
    """
    if baseline["scan_status"] != "completed":
        assert baseline["advisory_count"] is None
        assert baseline["verdict"] == "none_recorded"
        assert baseline["coverage_gap"] is True
        assert baseline["reason"], "a missing verdict has to say why"
    else:
        assert isinstance(baseline["advisory_count"], int)


def test_the_baseline_names_what_it_covered_with_its_denominator(baseline):
    """A count without the set it counted is how a 0 becomes a false clean.

    `advisory_count` is about these files and these distributions; the file
    carries the paths, per-file package counts and content hashes so a reader can
    re-derive the dependency set in the same breath.
    """
    covered = {row["path"]: row for row in baseline["dependency_set"]}
    assert "requirements.txt" in covered and "requirements.lock" in covered, covered
    for path, row in covered.items():
        assert (sc._repo_root() / path).is_file(), path
        assert row["packages"] > 0, f"{path} covers 0 packages"
        assert len(row["sha256"]) == 16


def test_the_baseline_is_tracked_by_git(baseline):
    """`eval/supply-chain/baseline.json` is ignored by the repo-wide `*.json` rule
    and `.gitignore` is a denied path for a round, so the extension is load-bearing:
    a JSON baseline would land un-tracked and invisible to every later reader."""
    path = sc.baseline_path()
    rel = path.relative_to(sc._repo_root())
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)],
                             cwd=str(sc._repo_root()), capture_output=True, text=True)
    assert tracked.returncode == 0, tracked.stderr
    ignored = subprocess.run(["git", "check-ignore", "-v", str(rel)],
                             cwd=str(sc._repo_root()), capture_output=True, text=True)
    assert ignored.returncode != 0, f"{rel} is gitignored: {ignored.stdout}"


# ── #1914: the three advisories this file used to carry, and the lock that clears them ──

#: id → (package, the version the advisory was reported against, first fixed version),
#: as the committed baseline recorded them before #1914 and as the mirrored OSV records
#: give the fix. Named so a cleared baseline cannot be satisfied by an empty file.
ADVISED_BEFORE_1914 = {
    "PYSEC-2026-3447": ("setuptools", "81.0.0", "83.0.0"),
    "GHSA-h35f-9h28-mq5c": ("setuptools", "81.0.0", "83.0.0"),
    "GHSA-rrmf-rvhw-rf47": ("torch", "2.11.0", "2.13.0"),
}


def _lock_pin(name: str) -> str:
    lock = sc._repo_root() / "requirements.lock"
    for raw in lock.read_text(encoding="utf-8").splitlines():
        pinned, sep, version = raw.strip().partition("==")
        if sep and pinned.strip().lower().replace("_", "-") == name:
            return version.strip()
    raise AssertionError(f"{name} is not pinned in requirements.lock")


def test_requirements_lock_pins_torch_and_setuptools_at_or_past_their_fixes():
    """The lock is the file the scan reads (`requirements.txt` names neither version), so
    this is the whole of "no advised version remains in the set the scan covers". It says
    nothing about what is INSTALLED — `tests/test_supply_chain_lock_pins.py` does."""
    from packaging.version import Version
    for advisory_id, (pkg, advised, fixed) in sorted(ADVISED_BEFORE_1914.items()):
        pin = Version(_lock_pin(pkg))
        assert pin != Version(advised), f"{advisory_id}: the lock still pins {pkg}=={advised}"
        assert pin >= Version(fixed), f"{advisory_id}: lock pins {pkg}=={pin}, below {fixed}"


def test_the_committed_baseline_is_a_scan_of_the_lock_now_in_the_tree(baseline):
    """`advisories: []` proves nothing alone. `dependency_set` carries a truncated sha256 of
    each file the scanner was handed; matching it against the bytes on disk is what says
    this verdict came out of the lock a reader can open."""
    import hashlib
    assert baseline["scan_status"] == "completed", baseline["scan_status"]
    covered = {row["path"]: row for row in baseline["dependency_set"]}
    assert "requirements.lock" in covered, sorted(covered)
    for rel, row in covered.items():
        digest = hashlib.sha256((sc._repo_root() / rel).read_bytes()).hexdigest()[:16]
        assert row["sha256"] == digest, (
            f"{rel}: baseline records {row['sha256']}, the file hashes to {digest} — "
            "regenerate with `python -m app.harness.supply_chain scan --write-baseline`")


def test_the_baseline_reports_no_advisory_naming_torch_or_setuptools(baseline):
    assert baseline["scan_status"] == "completed", baseline["scan_status"]
    named = {rec["package"].lower() for rec in baseline["advisories"]}
    assert not (named & {"torch", "setuptools"}), f"still flagged: {sorted(named)}"
    ids = {rec["id"] for rec in baseline["advisories"]}
    assert not (ids & set(ADVISED_BEFORE_1914)), sorted(ids & set(ADVISED_BEFORE_1914))
    assert baseline["advisory_count"] == len(baseline["advisories"])


# ── #2593 clause 1 — the baseline may only describe a tree a reader can open ──

def _report(**kw):
    """A completed, zero-advisory scan whose argv points at files of `root`."""
    base = kw.pop("root")
    files = [Path(base) / "requirements.txt", Path(base) / "requirements.lock"]
    return sc.ScanReport(
        scan_status="completed", scanner_name="osv-scanner", scanner_version="2.6.0",
        scanner_path="/usr/bin/true",
        argv=sc.OsvScannerSpec().argv(files),
        offline=True, db_directory="/tmp/db", db_freshness=_NOW_ISO,
        duration_seconds=1.0, advisory_count=0,
        dependency_set=[{"path": "requirements.txt", "packages": 1,
                         "sha256": "aaaa"}],
        generated_at=_NOW_ISO, **kw)


_NOW_ISO = "2026-10-11T12:00:00+00:00"


def test_write_baseline_refuses_a_target_outside_the_live_checkout(tmp_path):
    """The 2026-10-01 defect, planted: a scan run in a worktree, committed as a
    standing claim about the tree.

    `/home/alansrobotlab/lloyd-hand/G/` is the path the shipped artifact still
    names — a deleted round's worktree, which no one can go and look at, under a
    verdict that reads `no_advisories`. The check is on the *input*, not on where
    the YAML lands: a caller may legitimately redirect `path`, and that is not
    permission to describe another tree.
    """
    root = tmp_path / "live"
    root.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text("requests==2.32.0\n", encoding="utf-8")
    (root / "requirements.lock").write_text("requests==2.32.0\n", encoding="utf-8")
    dead = tmp_path / "lloyd-hand" / "G"
    dead.mkdir(parents=True)
    for name in ("requirements.txt", "requirements.lock"):
        (dead / name).write_text("requests==2.32.0\n", encoding="utf-8")

    report = _report(root=root)
    report.argv = ["scan", "--offline", "--format", "json",
                   "--lockfile", str(dead / "requirements.txt"),
                   "--lockfile", f"requirements.txt:{dead / 'requirements.lock'}"]
    out = tmp_path / "out" / "baseline.yaml"
    with pytest.raises(sc.BaselineTargetOutsideCheckout) as caught:
        sc.write_baseline(report, out, root=root)
    assert "lloyd-hand" in str(caught.value)
    assert not out.exists(), (
        "the artifact was written and then refused — a refused baseline still on "
        "disk is the lie this node exists to stop")
    assert not out.parent.exists() or list(out.parent.iterdir()) == []


def test_a_scan_target_may_be_a_path_the_box_has_but_the_checkout_does_not(tmp_path):
    """The other half of the rule, so the refusal cannot be widened into "only
    files under git": a target that exists but sits outside the live checkout is
    still unreviewable by the person reading the verdict."""
    root = tmp_path / "live"
    root.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text("requests==2.32.0\n", encoding="utf-8")
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    (elsewhere / "requirements.txt").write_text("x==1\n", encoding="utf-8")
    report = _report(root=root)
    report.argv = ["scan", "--lockfile", str(elsewhere / "requirements.txt")]
    offenders = sc.check_scan_targets(report.argv, root=root, live_root=root)
    assert offenders and "outside the live checkout" in offenders[0], offenders


def test_write_baseline_records_the_commit_of_the_tree_it_describes(tmp_path, monkeypatch):
    """The artifact names a path *and* a commit, so the claim survives the path.

    A worktree is deleted; a commit is not. `git show <sha>:requirements.lock` is
    how a reader checks a baseline whose tree is gone, which is exactly what
    `architecture/index.md` already does for retired docs and what the 2026-10-01
    artifact could not support because it recorded no sha at all.
    """
    root = tmp_path / "live"
    root.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text("requests==2.32.0\n", encoding="utf-8")
    (root / "requirements.lock").write_text("requests==2.32.0\n", encoding="utf-8")
    import subprocess
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["add", "-A"],
                 ["commit", "-q", "-m", "seed"]):
        subprocess.run(["git", "-C", str(root), *args], check=True,
                       capture_output=True)
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(sc, "_live_checkout", lambda: root)
    ran = _report(root=root)
    ran.tree = sc.tree_record(root)
    out = tmp_path / "baseline.yaml"
    sc.write_baseline(ran, out, root=root)
    text = out.read_text(encoding="utf-8")
    doc = yaml.safe_load(text[text.index("schema:"):])
    assert doc["tree"]["commit"] == sha
    assert doc["tree"]["files"] == ["requirements.txt", "requirements.lock"]
    argv_paths = sc.scan_target_paths(doc["scanner"]["argv"])
    assert argv_paths, "the artifact wrote no scan targets at all"
    for raw in argv_paths:
        assert Path(raw).is_file(), f"{raw} does not resolve on this box"
    assert "lloyd-hand" not in out.read_text(encoding="utf-8")
