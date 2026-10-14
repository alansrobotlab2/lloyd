"""Clauses 1 and 2 of #688: the offline advisory scan wrapper.

The seam under test is process-boundary sized on purpose — this module shells out
to a binary that is not installed on a round's tree (installing it is a person's
job: `.venvs/**` is a denied path and the advisory database is machine state). So
every test here runs a *stub scanner* placed on PATH, and asserts the flags the
real binary would have received. A test that only checked a returned dataclass
would pass whether or not the real invocation was ever offline.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from app.harness import supply_chain as sc

# What osv-scanner v2 prints for a lockfile scan with two findings, in the shape
# captured from real v2.6.0 `--format json` output on 2026-09-30 (#1838):
# `results[].packages[] = {"package": {name, version, ecosystem},
# "vulnerabilities": [...], "groups": [...]}`. The fixture used to model an
# invented `package_vulnerabilities` shape, which is how a parser that dropped
# every real finding passed.
OSV_REPORT = {
    "results": [
        {"source": {"path": "requirements.lock", "type": "lockfile"},
         "packages": [
             {"package": {"name": "mcp", "version": "2.0.0", "ecosystem": "PyPI"},
              "groups": [{"ids": ["GHSA-2q8f-6q6f-aaaa"]}, {"ids": ["PYSEC-2026-111"]}],
              "vulnerabilities": [
                  {"id": "GHSA-2q8f-6q6f-aaaa", "summary": "RCE in server",
                   "aliases": ["CVE-2026-0001"]},
                  {"id": "PYSEC-2026-111", "summary": "path traversal", "aliases": []}]},
             {"package": {"name": "httpx", "version": "0.27.0", "ecosystem": "PyPI"},
              "groups": [], "vulnerabilities": []},
         ]},
    ],
    "experimental_config": {},
}

#: One JSON object per invocation, appended. The wrapper invokes the scanner more
#: than once per scan (`--version` for the record, then the scan itself), so a
#: single-record file would only ever hold whichever call finished last.
STUB_TEMPLATE = """#!{python}
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "calls.jsonl"), "a") as fh:
    fh.write(json.dumps({{"argv": sys.argv[1:],
                         "db": os.environ.get("OSV_SCANNER_LOCAL_DB_CACHE_DIRECTORY"),
                         "api": os.environ.get("LLOYD_OSV_API_URL")}}) + "\\n")
sys.stdout.write({body!r})
"""


def _calls(bin_dir: Path) -> list[dict]:
    """Every scanner invocation the stub recorded, in order."""
    log = bin_dir / "calls.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def _scan_call(bin_dir: Path) -> dict:
    """The one call that is a scan, as opposed to the version probe."""
    scans = [c for c in _calls(bin_dir) if any("--offline" in str(a) for a in c["argv"])]
    assert scans, _calls(bin_dir)
    return scans[0]


def _install_stub(tmp_path: Path, body: str, *, name: str = "osv-scanner") -> Path:
    """Put an executable stub scanner on a PATH that contains only it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / name
    stub.write_text(STUB_TEMPLATE.format(python=sys.executable, body=body))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "requirements.txt").write_text("fastapi>=0.100\nmcp>=2\n")
    (repo / "requirements.lock").write_text("fastapi==0.104.1\nmcp==2.0.0\n"
                                            "httpx==0.27.0\n")
    return repo


@pytest.fixture(autouse=True)
def _no_real_scanner_or_service(monkeypatch):
    """Neither a stray `osv-scanner` on this box nor a live OSV service leaks in."""
    monkeypatch.delenv("LLOYD_OSV_API_URL", raising=False)
    monkeypatch.delenv("LLOYD_OSV_SCANNER_ARGS", raising=False)
    monkeypatch.delenv(sc.OSV_DB_ENV, raising=False)


def _isolate(tmp_path, monkeypatch, bin_dir):
    monkeypatch.setenv("PATH", str(bin_dir))
    db = tmp_path / "osv-db" / "osv"
    db.mkdir(parents=True, exist_ok=True)
    (db / "all.zip").write_bytes(b"zipzipzip")
    monkeypatch.setenv(sc.OSV_DB_ENV, str(tmp_path / "osv-db"))
    return tmp_path / "osv-db"


def test_stub_scanner_is_invoked_offline_with_a_local_database_and_the_dependency_set(
        tmp_path, monkeypatch):
    bin_dir = _install_stub(tmp_path, json.dumps(OSV_REPORT))
    db_dir = _isolate(tmp_path, monkeypatch, bin_dir)
    repo = _repo(tmp_path)

    report = sc.run_offline_scan(repo)
    recorded = _scan_call(bin_dir)

    assert report.scan_status == "completed", report.reason
    # Clause 1, part one: the no-network flag and the local database directory.
    assert "--offline" in recorded["argv"], recorded["argv"]
    assert recorded["db"] == str(db_dir), recorded["db"]
    # ...against the repo's declared dependency set, one lockfile flag each.
    assert "--lockfile" in recorded["argv"]
    lockfiles = [recorded["argv"][i + 1]
                 for i, tok in enumerate(recorded["argv"]) if tok == "--lockfile"]
    assert sorted(Path(p).name for p in lockfiles) == ["requirements.lock",
                                                       "requirements.txt"]
    assert report.offline is True
    assert report.scanner_name == "osv-scanner"
    assert report.argv == recorded["argv"]


def test_the_download_databases_staging_flag_is_never_passed(tmp_path, monkeypatch):
    """`--download-offline-databases` needs network and belongs to staging only.

    Passing it at scan time would make the wrapper's `offline: true` a lie on the
    exact run it is supposed to prove.
    """
    bin_dir = _install_stub(tmp_path, json.dumps(OSV_REPORT))
    _isolate(tmp_path, monkeypatch, bin_dir)

    report = sc.run_offline_scan(_repo(tmp_path))

    # Named flags, not a substring: this test's own tmp path carries the word.
    assert not any(tok.startswith("--download") for tok in report.argv), report.argv
    assert not any(tok.startswith("--download")
                   for call in _calls(bin_dir) for tok in call["argv"])


def test_the_stub_report_parses_into_advisories_with_their_ids(tmp_path, monkeypatch):
    bin_dir = _install_stub(tmp_path, json.dumps(OSV_REPORT))
    _isolate(tmp_path, monkeypatch, bin_dir)

    report = sc.run_offline_scan(_repo(tmp_path))

    assert report.advisory_count == 2, [a.__dict__ for a in report.advisories]
    assert {a.id for a in report.advisories} == {"GHSA-2q8f-6q6f-aaaa",
                                                 "PYSEC-2026-111"}
    assert {a.package for a in report.advisories} == {"mcp"}
    assert report.to_dict()["verdict"] == "advisories_found"


def test_a_completed_scan_records_version_duration_and_database_freshness(
        tmp_path, monkeypatch):
    """The four things clause 3 asks a baseline to carry, populated for real.

    The committed baseline on a tree with no scanner has these fields null and
    says so; this test is what proves the recording machinery itself works, so a
    null in the committed file is known to mean "no instrument", not "broken
    recorder".
    """
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    stub = tmp_path / "bin" / "osv-scanner"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\n"
        "if '--version' in sys.argv: sys.stdout.write('osv-scanner version 2.2.4\\n')\n"
        f"else: sys.stdout.write({json.dumps(json.dumps(OSV_REPORT))})\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    db_dir = _isolate(tmp_path, monkeypatch, tmp_path / "bin")

    report = sc.run_offline_scan(_repo(tmp_path))

    assert report.scanner_version == "2.2.4", report.scanner_version
    assert report.duration_seconds is not None and report.duration_seconds >= 0.0
    assert report.db_directory == str(db_dir)
    assert report.db_freshness is not None
    parsed = dt.datetime.fromisoformat(report.db_freshness)
    assert parsed.tzinfo is not None, "a naive freshness stamp reads as UTC later"
    assert report.advisory_count == 2


@pytest.mark.parametrize("body", ["", "not json at all", "{}", '{"results": null}',
                                  '{"unexpected": 1}'])
def test_a_scanner_body_that_is_not_a_report_is_a_failure_never_a_zero(
        tmp_path, monkeypatch, body):
    bin_dir = _install_stub(tmp_path, body)
    _isolate(tmp_path, monkeypatch, bin_dir)

    report = sc.run_offline_scan(_repo(tmp_path))

    assert report.scan_status == "failed", body
    assert report.advisory_count is None
    assert report.reason


def test_no_scanner_on_path_is_reported_as_the_missing_binaries(tmp_path, monkeypatch):
    """Clause 2: a non-zero exit naming the binary it looked for."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    spec, why = sc.resolve_scanner()
    report = sc.run_offline_scan(_repo(tmp_path))

    assert spec is None
    assert "osv-scanner: not on PATH" in why, why
    assert "pip-audit" in why, why
    assert report.scan_status == "scanner_absent"
    assert "osv-scanner" in report.reason and "pip-audit" in report.reason
    assert report.coverage_gap is True


def test_an_absent_scanner_never_yields_a_zero_advisory_verdict(tmp_path, monkeypatch):
    """The failure this wrapper exists to avoid: a clean tree from no instrument."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    report = sc.run_offline_scan(_repo(tmp_path))
    blob = report.to_dict()

    assert report.advisory_count is None, "an absent scanner must not report a count"
    assert blob["verdict"] == "none_recorded"
    assert blob["scan_status"] == "scanner_absent"
    assert "not a clean tree" in blob["reason"]


def test_the_absent_reason_names_the_release_route_and_never_the_registry(
        tmp_path, monkeypatch):
    """#1838: the old reason said `uv tool install osv-scanner`, a name PyPI does not
    publish and this module's own provenance check refuses. The reason now names the
    checksummed GitHub asset and a database path a reader can mkdir."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    reason = sc.run_offline_scan(_repo(tmp_path)).reason

    assert "uv tool install" not in reason and "pip install" not in reason
    assert "github.com/google/osv-scanner" in reason
    assert "osv-scanner_linux_amd64" in reason and "osv-scanner_SHA256SUMS" in reason
    assert "not published" in reason
    assert sc.OSV_DB_ENV in reason
    assert str(Path("supply-chain") / "osv-db") in reason


def test_an_unmirrored_database_is_a_gap_not_a_clean_scan(tmp_path, monkeypatch):
    """Measured 2026-09-30: real osv-scanner --offline over an empty database dir
    exits 0 with no findings. The stub does the same; the wrapper must not believe it."""
    bin_dir = _install_stub(tmp_path, json.dumps({"results": []}))
    monkeypatch.setenv("PATH", str(bin_dir))
    empty_db = tmp_path / "osv-db"
    empty_db.mkdir()
    monkeypatch.setenv(sc.OSV_DB_ENV, str(empty_db))

    report = sc.run_offline_scan(_repo(tmp_path))

    assert report.scan_status == "failed"
    assert report.advisory_count is None
    assert report.coverage_gap is True
    assert "no mirrored database" in report.reason and "not a clean tree" in report.reason
    assert report.to_dict()["verdict"] == "none_recorded"


def test_a_package_entry_of_an_unknown_shape_is_unreadable_not_zero():
    """The old parser read an invented `package_vulnerabilities` shape and counted a
    real report as 0. A package entry without upstream's `package` object is now a
    report the wrapper cannot read."""
    old_shape = {"results": [{"packages": [
        {"name": "mcp", "version": "2.0.0",
         "package_vulnerabilities": [{"osv_id": "GHSA-x", "summary": "s"}]}]}]}

    assert sc._parse_scan_output("osv-scanner", json.dumps(old_shape)) is None
    advisories, count = sc._parse_scan_output("osv-scanner", json.dumps(OSV_REPORT))
    assert count == 2
    assert {(a.package, a.version, a.ecosystem) for a in advisories} == {
        ("mcp", "2.0.0", "PyPI")}


def test_a_requirements_lock_is_handed_over_with_its_format_named():
    """osv-scanner picks a parser by file name and knows no `requirements.lock`:
    measured 2026-09-30 it exited 127 and the scan read as failed. The lock is
    `pip freeze` output, so it goes over as `requirements.txt:<path>`."""
    argv = sc.OsvScannerSpec().argv([Path("/r/requirements.txt"),
                                     Path("/r/requirements.lock"),
                                     Path("/r/requirements-dev.txt")])
    lockfiles = [argv[i + 1] for i, tok in enumerate(argv) if tok == "--lockfile"]

    assert lockfiles == ["/r/requirements.txt", "requirements.txt:/r/requirements.lock",
                         "/r/requirements-dev.txt"]


def test_the_cli_exits_non_zero_when_no_scanner_can_run(tmp_path, monkeypatch):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    assert sc.main(["scan"]) == 3, "a run with no scanner is not a success"


def test_pip_audit_is_refused_unless_its_service_url_is_a_local_mirror(tmp_path,
                                                                       monkeypatch):
    """pip-audit has no offline flag; its offline knob is a local OSV service URL.

    A URL pointing off-box would make network calls, so the spec reports the
    prerequisite unmet rather than run something that phones home and gets
    recorded as `offline: true`.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "pip-audit"
    stub.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write('[]')\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(bin_dir))

    spec, why = sc.resolve_scanner()
    assert spec is None
    assert sc.OSV_API_URL_ENV in why

    monkeypatch.setenv(sc.OSV_API_URL_ENV, "http://127.0.0.1:8089/osv")
    spec, why = sc.resolve_scanner()
    assert spec is not None and spec.name == "pip-audit", why
    argv = spec.argv([_repo(tmp_path) / "requirements.txt"])
    assert "-s" in argv and "osv" in argv
    assert "--osv-url" in argv and "http://127.0.0.1:8089/osv" in argv

    monkeypatch.setenv(sc.OSV_API_URL_ENV, "https://api.osv.dev/v1")
    assert sc.resolve_scanner()[0] is None, "a remote service is not an offline scan"


# ── #2593 clause 5 — a verdict has a shelf life, and says which clock ran out ──

NOW = dt.datetime(2026, 10, 11, 12, 0, tzinfo=dt.timezone.utc)


def _doc(generated_at: str, freshness: str) -> dict:
    return {"schema": sc.BASELINE_SCHEMA, "scan_status": "completed",
            "verdict": "no_advisories", "generated_at": generated_at,
            "database": {"directory": "/tmp/db", "freshness": freshness},
            "advisory_count": 0}


def _fresh(hours: float = 0.0) -> str:
    """Stamp `hours` before the frozen `NOW`, for a call that is *told* the clock.

    Every pure-function node below passes `now=NOW` to `stale_stamps` /
    `effective_verdict`, so the instant the age is judged at is part of the
    assertion and the node cannot rot. Do not use this helper for a `ScanReport`:
    see `_live`.
    """
    return (NOW - dt.timedelta(hours=hours)).isoformat()


def _live(hours: float = 0.0) -> str:
    """Stamp `hours` before the *real* clock, for a `ScanReport`.

    `ScanReport.verdict_label` is the one staleness consumer with no `now=`
    parameter (`app/harness/supply_chain.py`: it calls `stale_stamps` without
    passing one, which is correct — production builds the record at scan time,
    when the wall clock is the only right answer). So an age given to that object
    is measured against the day the suite runs, and a stamp built from the frozen
    `NOW` above is a fixed calendar date masquerading as an age.

    #2707 is what that costs: `test_a_scan_run_against_a_stale_mirror_reports_stale_not_clean`
    stamped its "fresh" control `_fresh(1)` — 2026-10-11 11:00, one hour before
    `NOW` — and asserted `no_advisories`. `STALE_GENERATED_MAX_DAYS` is 3, so that
    held until 2026-10-14 11:00Z and then said `stale` on every run, red in a gate
    report with no diff in sight of it. Anchored here, the same age means the same
    thing on every date the suite is ever run.
    """
    return (dt.datetime.now(dt.timezone.utc)
            - dt.timedelta(hours=hours)).isoformat()


def test_a_two_day_old_clean_scan_is_still_a_clean_scan():
    """The control on the other side of both ceilings: nothing is downgraded
    inside the shelf life, so the rule is a decay and not a blanket refusal that
    a tired operator switches off."""
    doc = _doc(_fresh(24), _fresh(24))
    assert sc.stale_stamps(doc, now=NOW) == []


def test_a_stale_generated_at_downgrades_the_verdict_and_names_the_stamp():
    """`generated_at` older than the verdict's life: the scan ran, and what it
    proved has an expiry date on it. Naming the stamp is what makes the downgrade
    actionable — "stale" alone does not say whether to re-scan or re-mirror."""
    doc = _doc(_fresh(4 * 24), _fresh(24))
    reasons = sc.stale_stamps(doc, now=NOW)
    assert len(reasons) == 1, reasons
    assert "generated_at" in reasons[0] and "4.0 days" in reasons[0], reasons
    assert sc.effective_verdict(doc, now=NOW) == sc.STALE_VERDICT
    assert "3-day" in reasons[0], reasons[0]


def test_a_stale_database_freshness_downgrades_even_a_fresh_scan():
    """The mirrored OSV database is a second, independent clock. A scan run this
    morning against a nine-day-old mirror is fresh paperwork over stale knowledge:
    OSV publishes new advisories against versions already in the tree, so the
    verdict decays with no change to the tree at all."""
    doc = _doc(_fresh(1), _fresh(9 * 24))
    reasons = sc.stale_stamps(doc, now=NOW)
    assert len(reasons) == 1 and "database.freshness" in reasons[0], reasons
    assert sc.effective_verdict(doc, now=NOW) == sc.STALE_VERDICT


def test_both_clocks_failing_names_both():
    doc = _doc(_fresh(30 * 24), _fresh(30 * 24))
    reasons = sc.stale_stamps(doc, now=NOW)
    assert len(reasons) == 2, reasons
    assert any("generated_at" in r for r in reasons)
    assert any("database.freshness" in r for r in reasons)


def test_the_shipped_baseline_of_2026_10_01_fails_both_ceilings():
    """The demonstration the item asked for, run against the artifact as it was
    committed rather than against a fiction: 10 days past the verdict's life and
    11 past the mirror's, presenting `no_advisories` the whole time."""
    doc = {"schema": sc.BASELINE_SCHEMA, "scan_status": "completed",
           "verdict": "no_advisories",
           "generated_at": "2026-10-01T14:30:00.898424Z",
           "database": {"directory": "/tmp/db",
                        "freshness": "2026-09-30T14:27:46.692030Z"},
           "advisory_count": 0}
    reasons = sc.stale_stamps(doc, now=NOW)
    assert len(reasons) == 2, reasons
    assert sc.effective_verdict(doc, now=NOW) == sc.STALE_VERDICT


def test_an_unreadable_stamp_counts_as_stale_never_as_fresh():
    """A stamp this cannot parse cannot earn a present tense by being ambiguous —
    which is the naive-local-timestamp bug in artifact form."""
    doc = _doc("last tuesday", "not-a-date")
    reasons = sc.stale_stamps(doc, now=NOW)
    assert any("generated_at" in r for r in reasons), reasons


def test_a_coverage_gap_is_never_relabeled_stale():
    """Only an artifact that claimed `no_advisories` decays. A scanner-absent
    record already says it produced no verdict, and turning that into `stale`
    would imply a scan that ran and aged — which is #688's honesty rule."""
    for verdict in ("none_recorded", "advisories_found"):
        doc = _doc(_fresh(90 * 24), _fresh(90 * 24))
        doc["verdict"] = verdict
        assert sc.stale_stamps(doc, now=NOW) == []
        assert sc.effective_verdict(doc, now=NOW) == verdict


def test_a_scan_run_against_a_stale_mirror_reports_stale_not_clean():
    """The edge a person hits: `run_offline_scan` builds the record, and the
    record's own verdict must not say `no_advisories` about a nine-day-old
    database. The report is the single definition — the printed line and the
    committed YAML both read `verdict_label`, so they cannot disagree."""
    report = sc.ScanReport(scan_status="completed", scanner_name="osv-scanner",
                           scanner_version="2.6.0", scanner_path="/usr/bin/true",
                           offline=True, db_directory="/tmp/db",
                           db_freshness=_live(9 * 24), duration_seconds=1.0,
                           advisory_count=0, generated_at=_live(1))
    assert report.verdict_label == "stale", report.verdict_label
    assert report.to_dict()["verdict"] == "stale"
    assert "database.freshness" in " ".join(report.to_dict()["stale"])
    assert "stale:" in sc._format_report(report)
    fresh = sc.ScanReport(scan_status="completed", scanner_name="osv-scanner",
                          scanner_version="2.6.0", scanner_path="/usr/bin/true",
                          offline=True, db_directory="/tmp/db",
                          db_freshness=_live(1), duration_seconds=1.0,
                          advisory_count=0, generated_at=_live(1))
    assert fresh.verdict_label == "no_advisories"
    assert fresh.to_dict()["stale"] == []
    assert "no_advisories" in sc._format_report(fresh)


def test_a_finding_is_not_downgraded_by_its_own_age():
    """Advisories present is a fact with a count beside it, not a clean claim: a
    stale scan that found something must keep saying it found something."""
    report = sc.ScanReport(scan_status="completed", scanner_name="osv-scanner",
                           scanner_version="2.6.0", scanner_path="/usr/bin/true",
                           offline=True, db_directory="/tmp/db",
                           db_freshness=_live(40 * 24), duration_seconds=1.0,
                           advisory_count=3, generated_at=_live(40 * 24))
    assert report.verdict_label == "advisories_found"
    assert report.to_dict()["advisory_count"] == 3


_FROZEN_HELPER = "_fresh"
_LIVE_CLOCK_BUILDER = "ScanReport"
_ALIAS_DEPTH = 4


def _frozen_stamp_findings(source: str) -> tuple[int, list[str]]:
    """Count `ScanReport` constructions in `source`; report any stamped from `_fresh`.

    Returns `(built, findings)`, a finding being `"line N: keyword 'x'"` naming the
    construction and the keyword that carries it.

    The first round of #2707 shipped this as an inline walk that recognised only a
    `_fresh(...)` call written *directly* inside a keyword, while its own docstring
    claimed a local alias was still caught. The review rung's mutation —
    `_aliased = _fresh(1)`, then `ScanReport(generated_at=_aliased)` — came back
    `built=1 frozen=[]`, green against the exact violation the check exists to refuse.
    So the walk carries one lexical scope at a time and resolves names through it.

    Followed: a `_fresh(...)` call anywhere inside a keyword's value; a name that an
    assignment statement in the construction's own scope, or in any enclosing function
    or module scope, binds to such a call — through *any* of its bindings, because a
    name with one frozen binding and one live binding can still carry the frozen one,
    and that branch rots on the same date; and a name bound to another such name, to a
    depth of `_ALIAS_DEPTH` assignments. A construction written in a nested function's
    decorators, annotation or argument defaults is walked too, against the enclosing
    scope's names.

    Not followed, stated so the next reader is not told a falsehood: a value that
    reaches the keyword through another function's return value, through an attribute,
    or through a name no assignment statement binds at all (a `for` or `with` target, a
    comprehension variable, a parameter's bound value). A name this walk cannot resolve
    is left alone rather than guessed at, and
    `test_the_frozen_clock_guard_catches_what_it_claims_to` pins both halves of that
    sentence.
    """
    tree = ast.parse(source, filename="<frozen-stamp-scan>")
    scopes_of = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def outside_body(node):
        """The parts of a def or class that are not its body: still the outer scope's."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return (*node.decorator_list, node.returns,
                    *node.args.defaults, *node.args.kw_defaults)
        return tuple(getattr(node, "decorator_list", ()))

    def descend(node):
        """`node` and its descendants, stopping at the body of a nested scope."""
        yield node
        if isinstance(node, scopes_of):
            for part in outside_body(node):
                yield from descend(part)
            return
        for child in ast.iter_child_nodes(node):
            yield from descend(child)

    def statements(stmts):
        """Nodes belonging to this scope: its statements plus their non-scope parts."""
        for stmt in stmts:
            yield stmt
            if isinstance(stmt, scopes_of):
                continue
            for child in ast.iter_child_nodes(stmt):
                yield from descend(child)

    def bindings(stmts) -> dict:
        """Name -> every value an assignment statement in this scope gives it."""
        out: dict[str, list] = {}
        for node in statements(stmts):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                targets, value = [node.target], node.value
            else:
                continue
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(value)
        return out

    def holds_frozen(expr, scopes, depth: int = 0) -> bool:
        if depth > _ALIAS_DEPTH:
            return False
        for inner in ast.walk(expr):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == _FROZEN_HELPER):
                return True
            if isinstance(inner, ast.Name) and name_holds_frozen(inner.id, scopes, depth + 1):
                return True
        return False

    def name_holds_frozen(name: str, scopes, depth: int) -> bool:
        if depth > _ALIAS_DEPTH:
            return False
        for scope in reversed(scopes):  # innermost scope wins: an outer binding is shadowed
            values = scope.get(name)
            if values:
                return any(holds_frozen(v, scopes, depth + 1) for v in values)
        return False

    built = 0
    findings: list[str] = []

    def visit(stmts, scopes) -> None:
        nonlocal built
        scopes = scopes + [bindings(stmts)]
        for node in statements(stmts):
            if isinstance(node, scopes_of):
                visit(node.body, scopes)
            elif (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == _LIVE_CLOCK_BUILDER):
                built += 1
                for keyword in node.keywords:
                    if holds_frozen(keyword.value, scopes):
                        findings.append(f"line {node.lineno}: keyword {keyword.arg!r}")

    visit(tree.body, [])
    return built, findings


def test_no_scan_report_in_this_file_is_stamped_from_a_frozen_clock():
    """#2707's class rule: a record judged by the wall clock gets a wall-clock stamp.

    `ScanReport.verdict_label` has no `now=` parameter and correctly so — production
    builds the record at scan time, when the real clock is the only right answer. So
    a `ScanReport` stamped from `_fresh(...)`, which is anchored to the frozen `NOW`
    above, carries not an age but a fixed calendar date, and that date's age grows by
    one day every day until it crosses a ceiling and the node begins asserting the
    opposite of what it means.

    That is exactly how main went red: the stale-mirror node stamped its *fresh*
    control `_fresh(1)` = 2026-10-11 11:00 and asserted `no_advisories`; the ceiling
    is 3 days, so at 2026-10-14 11:00Z the answer became `stale` and stayed there on
    every run, with no diff anywhere near it. Its siblings that call `stale_stamps`
    with `now=NOW` stayed green because they state the instant they are judged at —
    the rot was never the frozen constant, it was the frozen constant meeting a
    function that ignores it.

    The check is the AST, not a grep, so prose mentioning the helper is not a finding;
    `_frozen_stamp_findings` states in its own docstring exactly which carry shapes it
    follows and which it does not, and the node below tests that sentence rather than
    leaving it as a claim.
    """
    built, frozen = _frozen_stamp_findings(Path(__file__).read_text(encoding="utf-8"))
    assert built >= 3, (
        f"only {built} ScanReport constructions in this file: its subject moved, so "
        "a green here would prove nothing about the constructions that exist")
    assert not frozen, (
        f"{frozen} — a live record stamped from the frozen `NOW` is an age that "
        "rots with the calendar; use `_live` (#2707)")


def test_the_frozen_clock_guard_catches_what_it_claims_to():
    """The guard's own reach, measured on sources that are not this file.

    #2707's first round passed a claim about coverage it had not measured and the review
    rung broke it with a two-line mutation. So each form the docstring names is written
    out here and handed to `_frozen_stamp_findings`, and two forms it names as NOT
    followed are pinned too — a limitation stated in a docstring nobody tests is how the
    last round arrived.

    This is what makes the guard above falsifiable: an assertion that only ever sees one
    file cannot show it would have reddened on the red.
    """

    def build(stamp: str) -> str:
        return ("import app\n"
                f"report = app.ScanReport(scan_status='completed', generated_at={stamp})\n")

    # The shape that rotted, written directly, and the alias the review rung used.
    assert _frozen_stamp_findings(build("_fresh(1)")) == (
        1, ["line 2: keyword 'generated_at'"])
    aliased = ("import app\n"
               "_aliased = _fresh(1)\n"
               "report = app.ScanReport(scan_status='completed', generated_at=_aliased)\n")
    assert _frozen_stamp_findings(aliased) == (1, ["line 3: keyword 'generated_at'"]), (
        "a `_fresh` result reaching `ScanReport` through a function-local alias is the "
        "evasion #2707's first round shipped; it must not be green")

    # A chained alias, and a name whose several bindings include a frozen one.
    chained = ("import app\n"
               "first = _fresh(1)\n"
               "second = first\n"
               "report = app.ScanReport(generated_at=second)\n")
    assert _frozen_stamp_findings(chained) == (1, ["line 4: keyword 'generated_at'"])
    reassigned = ("import app\n"
                  "stamp = _live(1)\n"
                  "stamp = _fresh(1)\n"
                  "report = app.ScanReport(generated_at=stamp)\n")
    assert _frozen_stamp_findings(reassigned) == (1, ["line 4: keyword 'generated_at'"])

    # The live-clock control: same shape, right helper, no finding.
    assert _frozen_stamp_findings(build("_live(1)")) == (1, []), (
        "a guard that reports every construction is not a guard; it will be switched off")
    # `_fresh` is still correct for a call that is *told* the clock, so it must not be flagged.
    told = ("import app\n"
            "out = app.stale_stamps(app._doc(_fresh(1), _fresh(1)), now=app.NOW)\n")
    assert _frozen_stamp_findings(told) == (0, []), (
        "`_fresh` staying in this file is the point of `_live`; flagging its correct use "
        "is how a check gets deleted")

    # The two carry shapes the docstring states are NOT followed. If the walk grows to
    # follow one, this reddens and that sentence goes with it — do not extend the walk
    # and quietly delete this assertion, and never extend the claim without the reach.
    through_return = ("import app\n"
                      "def stamp():\n"
                      "    return _fresh(1)\n"
                      "report = app.ScanReport(generated_at=stamp())\n")
    assert _frozen_stamp_findings(through_return)[1] == [], (
        "the docstring says a value carried back from another function is not followed; "
        "it now is — update the sentence, do not drop this node")
    through_attribute = ("import app\n"
                         "class C:\n"
                         "    at = _fresh(1)\n"
                         "report = app.ScanReport(generated_at=C.at)\n")
    assert _frozen_stamp_findings(through_attribute)[1] == [], (
        "the docstring says an attribute is not followed; it now is — update the sentence")


def test_a_frozen_stamp_fails_the_very_ceiling_that_made_main_red():
    """The diagnosis kept as a witness, so the fix cannot be mistaken for a hush.

    `_fresh` stays in this file and stays right for every node that passes `now=NOW`;
    what it can no longer do is stamp a judgement made *today*. This node hands the
    live-clock reader the exact stamp of #2707's red run and requires `stale` — the
    failure reproduced rather than asserted.

    That stamp is written as a literal, not as `_fresh(1)`, for two reasons. It is
    monotone: 2026-10-11 11:00Z is `NOW` at `ff12b4d0` minus one hour, and a past
    instant only gets further past the ceiling, so this node cannot rot the way the
    node it documents did. And it does not couple to `NOW`: advancing a frozen fixture
    that eventually needs advancing would otherwise redden a supply-chain node with
    the same unattributed surprise #2707 arrived carrying.
    """
    incident_instant = "2026-10-11T11:00:00+00:00"
    generated = dt.datetime.fromisoformat(incident_instant)
    age_days = ((dt.datetime.now(dt.timezone.utc) - generated)
                .total_seconds() / 86400.0)
    assert age_days > sc.STALE_GENERATED_MAX_DAYS, (
        f"the #2707 stamp is {age_days:.2f} days old against the real clock, at or "
        f"under the {sc.STALE_GENERATED_MAX_DAYS}-day ceiling: either the ceilings "
        "moved or this box's clock went backwards — re-read this node by hand either "
        "way, it is not a supply-chain finding")
    doc = _doc(incident_instant, incident_instant)
    assert sc.effective_verdict(doc) == sc.STALE_VERDICT, (
        "a live-clock read of a stamp past its ceiling came back clean, which is the "
        "regression #2593 clause 5 exists to prevent")


# ── #2707 clause 2 — a green bought by not running is not a green ──

# The selection the automod gate runs this file under, spelled the way the gate spells
# it (`scripts/automod/gate.py`). A node that is only passing because this expression
# dropped it is not passing.
GATE_SELECTION = "not live_vault and not fault_injection"

# Marks whose effect is to remove an assertion rather than evaluate it. `live_vault`
# and `fault_injection` achieve that from outside the run, through the selection above;
# `skip`, `skipif` and `xfail` achieve it inside one.
_AWAY_MARKS = {"skip", "skipif", "xfail", "live_vault", "fault_injection"}

# The node #2707 was filed for, by name. Clause 2 is about THIS node going green by
# running, so it is named rather than described.
RED_AT_FILING = "test_a_scan_run_against_a_stale_mirror_reports_stale_not_clean"


def _test_functions() -> dict:
    """This file's top-level `test_*` functions, by name, with their decorators."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    return {node.name: node.decorator_list
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def _mark_names(decorators: list) -> set:
    """The mark names a decorator list applies, for `@pytest.mark.X` and `@pytest.mark.X(...)`.

    Resolved as `pytest.mark.<name>` on purpose: a node could be decorated some other
    way and still not run, but every spelling that reaches a mark goes through this
    attribute chain, and inventing a looser reader is how an inspection quietly passes
    over a form it does not understand.
    """
    names = set()
    for dec in decorators:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute):
            inner = target.value
            if inner.attr == "mark" and isinstance(inner.value, ast.Name) \
                    and inner.value.id == "pytest":
                names.add(target.attr)
    return names


def test_nothing_in_this_file_buys_its_green_by_not_running():
    """Clause 2, statically: this file has no escape hatch left in it.

    #2707's acceptance names four ways a red suite is made to look green — skipped,
    xfailed, deleted, marked `live_vault`. Marks and runtime `pytest.skip`/`xfail` calls
    are what this node can see, so it asserts all three absent, and it asserts the filed
    node still exists by name because a deleted assertion is invisible to every rule that
    reads only the surviving text.

    Complementary to the selection node below rather than redundant: a `skipif` mark
    still collects, so the measured node cannot see one and this one can, while a mark
    applied from a conftest or a `pytestmark` assembled in a loop reaches that one and
    could evade a per-function read. Verified by mutation both ways — a
    `@pytest.mark.skipif(True, …)` on any node reddens this one alone, and a
    `@pytest.mark.live_vault` reddens both.

    The one shape neither node can catch is a mark on the whole module, which would
    deselect this file including its own guards. The gate's skip rail does not catch it
    either (`PYTEST_MAX_SKIPPED = 40`, `scripts/automod/gate.py:118`: a deselected node
    is not a skipped one), so that shape belongs to diff review rather than to a suite
    assertion. Stated here instead of claimed as covered.

    The named node is checked for carrying no decorator at all, not merely none of
    `_AWAY_MARKS`: the fix for a red assertion is the assertion, and a marker of any
    kind on the node that was red is the wrong shape of change.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    module_marks = [node for node in tree.body
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and "pytestmark" in ast.dump(node)]
    assert not module_marks, (
        f"a module-level `pytestmark` ({[getattr(n.targets[0], 'id', '?') for n in module_marks]}) "
        "applies a mark to every node in the file, which is the one skip shape an "
        "assertion about individual functions cannot see")

    marked = {name: _mark_names(decs) & _AWAY_MARKS
              for name, decs in _test_functions().items()}
    marked = {name: hits for name, hits in marked.items() if hits}
    assert not marked, (
        f"{sorted(marked.items())} — a node here that declines to run is a clause "
        "recorded as satisfied with no evidence behind it (#2707 clause 2)")

    calls = sorted({f"{node.func.attr}" for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pytest"
                    and node.func.attr in {"skip", "xfail"}})
    assert not calls, (
        f"{calls} at runtime removes a node from a run that claims to have made it pass")

    functions = _test_functions()
    assert RED_AT_FILING in functions, (
        f"{RED_AT_FILING} is gone from this file. Deleting the node that was red is "
        "clause 2's named failure mode, and a green suite would not mention it")
    assert not functions[RED_AT_FILING], (
        f"{RED_AT_FILING} carries "
        f"{[ast.unparse(d) for d in functions[RED_AT_FILING]]}: the node filed red must "
        "be fixed, not marked")


def test_the_gate_selection_keeps_every_node_in_this_file():
    """Clause 2, measured: the selector the gate uses collects this file whole.

    The node above reads this file's text; this one asks pytest what it would actually
    run, twice — once with no selection and once with `GATE_SELECTION` verbatim — and
    requires the same node set both times. That is the only check here that can see the
    difference between "no marker is spelled in this file" and "nothing is excluded",
    because it goes through the expression rather than mirroring it: a mark applied from
    a conftest, a hook, or a `pytestmark` built by a loop all reach it.

    Run out of process against `--collect-only`, which imports and lists without
    executing: the inner run therefore cannot recurse into this node, and it costs the
    import rather than the suite. `PYTEST_ADDOPTS` is cleared so nothing in
    `pytest.ini` or the environment can widen the inner selection, and
    `-p no:cacheprovider` keeps it from leaving a `.pytest_cache` in a round's worktree.
    """
    file_path = str(Path(__file__).resolve())

    def collect(*selection: str) -> set:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", file_path, "--collect-only", "-q",
             "-p", "no:cacheprovider", *selection],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
            env={**{k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"},
                 "PYTEST_ADDOPTS": ""})
        assert proc.returncode == 0, (
            f"`pytest --collect-only` {selection or ()} exited "
            f"{proc.returncode}, so the comparison below has no denominator: "
            f"{proc.stdout[-500:]}{proc.stderr[-500:]}")
        # `-q --collect-only` prints one node id per line and nothing else that
        # contains `::`, so the separator is the whole filter.
        return {line.strip() for line in proc.stdout.splitlines() if "::" in line}

    unselected = collect()
    selected = collect("-m", GATE_SELECTION)

    assert unselected, "collection found no nodes at all, which is not a pass"
    assert selected == unselected, (
        f"the gate's selection drops {sorted(unselected - selected)} from this file: "
        f"a node excluded by `-m {GATE_SELECTION!r}` cannot satisfy clause 2")
    assert any(node.endswith(f"::{RED_AT_FILING}") for node in selected), (
        f"{RED_AT_FILING} is not in what the gate collects, so the clause-1 green the "
        "reviewer ran is not the green this suite reports")
