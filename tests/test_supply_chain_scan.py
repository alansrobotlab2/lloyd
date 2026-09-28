"""Clauses 1 and 2 of #688: the offline advisory scan wrapper.

The seam under test is process-boundary sized on purpose — this module shells out
to a binary that is not installed on a round's tree (installing it is a person's
job: `.venvs/**` is a denied path and the advisory database is machine state). So
every test here runs a *stub scanner* placed on PATH, and asserts the flags the
real binary would have received. A test that only checked a returned dataclass
would pass whether or not the real invocation was ever offline.
"""

from __future__ import annotations

import datetime as dt
import json
import stat
import sys
from pathlib import Path

import pytest

from app.harness import supply_chain as sc

# What osv-scanner v2 prints for a lockfile scan with two findings. The shape is
# the upstream `--format json` document: `results[].packages[]
# .package_vulnerabilities[]`.
OSV_REPORT = {
    "results": [
        {"type": "lockfile", "source": {"path": "requirements.lock"},
         "ecosystem": "PyPI",
         "packages": [
             {"name": "mcp", "version": "2.0.0",
              "package_vulnerabilities": [
                  {"osv_id": "GHSA-2q8f-6q6f-aaaa", "summary": "RCE in server"},
                  {"osv_id": "PYSEC-2026-111", "summary": "path traversal"}]},
             {"name": "httpx", "version": "0.27.0", "package_vulnerabilities": []},
         ]},
    ],
    "message": None,
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
