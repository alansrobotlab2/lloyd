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
