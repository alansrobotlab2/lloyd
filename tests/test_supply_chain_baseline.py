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


#: The tree these nodes read, named by this file's own location rather than by
#: `sc.baseline_path()`. That helper's root is `_repo_root()`, whose only fallback is
#: `app.paths.LLOYD_HOME` — `app/paths.py`'s own parent — so the answer follows
#: whichever `app` package the importing process loaded: the round's worktree under the
#: gate, the production checkout when the backend imports it. (`_live_checkout()` is a
#: different helper, the write guard's root, and is not on this read path.) A root that
#: depends on the import shape is exactly right for the shipped loader; it is the wrong
#: shape for a node whose job is to redden on this diff's bytes, because a node that
#: happens to resolve to production cannot see the artifact the round actually wrote.
#: #2593's defect was an artifact read against the wrong tree, so the #3613 nodes below
#: read the bytes beside this file and nothing else.
REPO_UNDER_TEST = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_UNDER_TEST / "eval" / "supply-chain" / "baseline.yaml"


@pytest.fixture(scope="module")
def artifact() -> dict:
    """The committed artifact's own bytes, parsed straight — no loader opinion, no
    root resolution, so the assertions are about what a reviewer can `git show`."""
    assert ARTIFACT.is_file(), (
        f"{ARTIFACT} is not readable, so the #3613 clauses below are ungraded")
    return yaml.safe_load(ARTIFACT.read_text(encoding="utf-8"))


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


def test_the_baseline_carries_none_of_the_advisories_1914_cleared(baseline):
    """#1914's promise, restated over the ids it cleared rather than over package names.

    This node used to assert that no advisory in the artifact named `torch` **or**
    `setuptools`. That held of the 2026-10-11 scan only because that scan ran against a
    mirror frozen at `2026-09-30T14:27:46.692030Z`: the refreshed mirror reports a
    *different* setuptools advisory — `GHSA-xmj3-9gf7-h52w`, "Windows path escape via
    namespace_packages.txt during legacy wheel installation", against `setuptools`
    `83.0.0`, which is verbatim what `requirements.lock:154` pins. #3613's own
    acceptance is that the declared half reports what the current mirror finds, so a
    blanket name-ban and the acceptance cannot both survive.

    What the body below enforces, and all of it: the three ids in `ADVISED_BEFORE_1914`
    are absent, no advisory names `torch`, and `advisory_count` equals the list's length.
    No advisory id or package is *allowed* by id, so a second setuptools finding would
    pass here too — stated rather than implied, because the ruling on which setuptools
    advisory this tree accepts is #3613's owed entry (upgrade it in the lock, or record
    the accepted floor beside #3614's model-serving one) and a test may not make that
    ruling in advance. The setuptools advisory the refresh surfaced is therefore not a
    regression of #1914, and this node is not the instrument that would clear it.
    """
    assert baseline["scan_status"] == "completed", baseline["scan_status"]
    ids = {rec["id"] for rec in baseline["advisories"]}
    assert not (ids & set(ADVISED_BEFORE_1914)), sorted(ids & set(ADVISED_BEFORE_1914))
    named = {rec["package"].lower() for rec in baseline["advisories"]}
    assert "torch" not in named, f"torch is flagged again: {sorted(named)}"
    assert baseline["advisory_count"] == len(baseline["advisories"])


# ── #3613: each half of the artifact has to name the reading that produced it ─────

def test_the_observed_half_names_the_reading_it_came_from(artifact):
    """Clause 1: the observed half was numbers without a provenance, so it aged out loud.

    `observed_generated_at` is the top-level stamp `write_merged_baseline` records for
    the observed half (`app/harness/supply_chain_inventory.py:903`) and the mirror stamp
    is a key of the `observed:` block itself — `attach_to_baseline` (`:862`) copies every
    observed key except `schema` and `generated_at`, so `observed_scan`'s own
    `database_freshness` (`:573`) arrives beside the counts rather than overwriting the
    declared `database.freshness`. The 2026-10-11 artifact carried neither next to its
    698 components and 150 missed advisories, which is how a reading of a different box
    passed as a current one for 20 days.

    The load-bearing assertion is relational, not a date: both halves came out of one
    process against one mirror, so they must agree on the mirror's stamp and be stamped
    minutes apart. An artifact that re-stamps one half and reuses the other is the shape
    this item was filed over, and no count in it is current.
    """
    obs = artifact.get("observed")
    assert isinstance(obs, dict), (
        "the artifact has no observed: half, so there is nothing to name")
    assert obs["scan_status"] == "completed", obs["scan_status"]

    stamp = artifact.get("observed_generated_at")
    assert stamp, (
        "observed_generated_at is absent: the observed numbers say nothing about when "
        "they were taken")
    observed_at = dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    assert observed_at.utcoffset() == dt.timedelta(0), f"{stamp} is not offset-bearing UTC"
    declared_at = dt.datetime.fromisoformat(
        artifact["generated_at"].replace("Z", "+00:00"))
    assert abs(observed_at - declared_at) <= dt.timedelta(hours=1), (
        f"the halves were stamped {observed_at} and {declared_at} — over an hour apart, "
        "so one half is a reused reading rather than this run's")

    mirror = obs.get("database_freshness")
    assert mirror, (
        "observed.database_freshness is null or absent: the missed-advisory count "
        "cannot be attributed to a mirror, which is the whole comparison #3544 added "
        "the key for")
    assert mirror == artifact["database"]["freshness"], (
        f"the observed half scanned mirror {mirror} while the declared half scanned "
        f"{artifact['database']['freshness']} — two eras stitched into one artifact")

    for key in ("component_count",
                "advisories_in_observed_set_that_declared_lockfiles_missed"):
        value = obs.get(key)
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"observed.{key} is {value!r}, not an integer count")
    assert (obs["component_count"],
            obs["advisories_in_observed_set_that_declared_lockfiles_missed"]) != (698, 150), (
        "the observed half still carries the 2026-10-11 reading of 698 components and "
        "150 missed advisories, which is the pair this item was filed to replace")


def _declared_half_violations(doc: dict) -> list[str]:
    """Every way a declared half can attest a tree it did not measure, as plain messages.

    One measurement, two directions: the node over the committed artifact and the node over
    seeded bytes both call this, so "it would catch that" and "it does not catch a clean
    rescan" are statements about one code path rather than about a docstring. Violations are
    returned rather than raised so a caller can ask the negative question.

    What counts as dishonest here is self-contradiction inside the artifact: a verdict the
    bytes do not carry, a count the list does not fill, a scan that did not finish while
    reporting findings. How many advisories the mirror finds is a fact about the box and
    never a violation — nothing in this function requires `advisory_count > 0`. That is
    clause 5 of #3613, and precisely the assertion the review rung refused twice: it made a
    future lock upgrade that clears the current findings redden a permanent witness.
    """
    problems: list[str] = []
    if doc.get("scan_status") != "completed":
        problems.append(f"scan_status is {doc.get('scan_status')!r}: a declared half "
                        "reports findings only if the scan finished")
    advisories = doc.get("advisories")
    if not isinstance(advisories, list):
        problems.append(f"advisories is {advisories!r}, not a list of findings")
        advisories = []
    count = doc.get("advisory_count")
    if count != len(advisories):
        problems.append(f"advisory_count {count!r} disagrees with the "
                        f"{len(advisories)} advisories the artifact itself carries")
    verdict = doc.get("verdict")
    if verdict == "stale":
        problems.append(f"verdict is 'stale' with stale: {doc.get('stale')!r} — the artifact "
                        "is condemning its own stamps, so neither its counts nor its "
                        "emptiness describe anything current")
    elif doc.get("stale") != []:
        problems.append(f"verdict {verdict!r} is paired with a non-empty stale: "
                        f"{doc.get('stale')!r}")
    if verdict == "advisories_found" and not advisories:
        problems.append("verdict is 'advisories_found' beside an empty advisories list: the "
                        "finding the verdict names is not in the file, and `verdict_label` "
                        "sets that verdict only from advisories it was handed")
    return problems


def test_the_declared_half_attests_only_the_advisories_the_mirror_found(artifact):
    """Clause 2: `advisory_count: 0` was a clean-tree claim the scan never made.

    The 2026-10-11 artifact recorded `advisory_count: 0` / `advisories: []` against a mirror
    frozen at 2026-09-30 and read `verdict: stale` — stale *because* the mirror was too old
    to trust the zero, which is what its `stale:` list said. Scanning the same three tracked
    files against the refreshed mirror finds advisories, so `verdict_label` returns that
    finding rather than a staleness verdict — it reaches `stale_stamps` only for a scan that
    came back clean — and the `stale:` list comes back empty. The regenerated half is
    therefore recomputed and not hand-edited: every assert here reads the bytes, so the only
    route to green is a scan whose verdict the bytes support.

    What this node deliberately does not assert: any count. The 30 advisories this box
    currently finds are stated in the skill and ruled on by #3613's owed entry (upgrade in
    the lock, or record the accepted floor); freezing them here as a floor is what
    `test_a_clean_rescan_and_a_hollow_one_divide_at_the_declared_half_node` exists to refuse.
    """
    assert artifact["scan_status"] == "completed", artifact["scan_status"]
    assert artifact["coverage_gap"] is False, (
        "coverage_gap is true: the declared half is counting a set it did not cover")
    problems = _declared_half_violations(artifact)
    assert not problems, f"the declared half does not attest what it measured: {'; '.join(problems)}"


def test_a_clean_rescan_and_a_hollow_one_divide_at_the_declared_half_node():
    """Clause 5: the declared-half rule has to be able to say "honest, and empty".

    The refused version of `test_the_declared_half_attests_only_the_advisories_the_mirror_found`
    ended in an unconditional `assert artifact["advisory_count"] > 0 and advisories` while its
    docstring promised "an honestly clean future rescan still passes" — a witness that goes red
    for good the day somebody upgrades the pins the mirror is advising against, and whose prose
    said the opposite of its body. Clause 5 closes that by moving the question into a rule over
    four seeded artifacts, three of which must be refused and one accepted.

    Seeded rather than read from disk because the accepted case has to be true of a rule while
    it is not true of the box: today's mirror does find advisories.
    """
    def seed(**over) -> dict:
        doc = {"scan_status": "completed", "verdict": "advisories_found",
               "advisory_count": 1, "advisories": [{"id": "GHSA-seeded-example"}],
               "stale": []}
        doc.update(over)
        return doc

    assert _declared_half_violations(seed()) == [], (
        "the rule refuses an artifact carrying one advisory in a list of one")

    clean = seed(verdict="no_advisories", advisory_count=0, advisories=[])
    assert _declared_half_violations(clean) == [], (
        "an honestly clean rescan is being refused — the brittleness clause 5 removes, and "
        "what a lock upgrade that clears every finding would hit")

    hollow = seed(verdict="advisories_found", advisory_count=0, advisories=[])
    assert _declared_half_violations(hollow), (
        "a verdict of advisories_found beside an empty advisories list passed the rule")

    aged = seed(verdict="stale", advisory_count=0, advisories=[],
                stale=["database.freshness 2026-09-30T14:27:46.692030Z is past its rail"])
    assert _declared_half_violations(aged), (
        "the 2026-10-11 shape — a zero its own staleness list had already condemned — now "
        "satisfies the rule this item was filed to write")

    unfinished = seed(scan_status="failed")
    assert any("scan_status" in p for p in _declared_half_violations(unfinished)), (
        "a scan that did not complete is invisible to the rule")


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
