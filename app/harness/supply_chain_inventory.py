"""The observed component inventory behind the supply-chain verdict (#2593).

Why this module exists
----------------------
`app/harness.supply_chain` scans three declared lockfiles and commits the result
as a standing fact. Three things were wrong with that, all measured on this box
on 2026-10-11: the committed `scanner.argv` named
`/home/alansrobotlab/lloyd-hand/G/requirements{.txt,.lock,-dev.txt}`, a
worktree that no longer exists, so the artifact attesting `no_advisories` was
produced by a command nobody can re-run; nothing on the box regenerates it, so
`generated_at: 2026-10-01` / `database.freshness: 2026-09-30` sat presenting a
present-tense `no_advisories` at 10 and 11 days old; and the target is one
environment's *declared* set — `requirements.lock`, 180 names — while the box
runs five more interpreters whose installed sets no file describes.

This module answers the second half of that. It enumerates what is actually
installed, feeds it to the offline osv-scanner path that already ships, and
reports the advisories the observed set has that the declared lockfiles do not,
as its own number.

The scope, decided here rather than deferred
--------------------------------------------
#688's triage ended with "the scan target must be decided, not assumed". The
decision: **every site-packages a service launcher or the supervisor config
execs, plus every `~/.local/lib/python3.*/site-packages`** — the importable set
of the processes that run, not a disk census. Which venvs that means is derived
from the files that name them (`SCOPE_FILES`), not from a list: a venv created
for a new engine is in the inventory the first time it is in a launcher, and a
venv whose launcher loses its `VIRTUAL_ENV` line drops out. `.gitignore:3`
excludes `.venvs/`, so the checkout cannot carry that list and remain truthful
about it.

On 2026-10-11 that resolves to five venvs plus the 3.14 user-site — 960
component rows, 511 distinct name+version pairs. The declared lockfile set
(181 names) misses installed distributions in every one of the four engine
venvs: 28 (qwen3-tts), 99 and 146 (the vllm trees).

What this is NOT
----------------
Not a remediation instrument. It reports where a running interpreter has a
vulnerable package and installs nothing; the 139 observed-but-not-declared
advisories it measured on 2026-10-11 are Alan's standing decision, not this
round's. Not a new scanner either: it synthesises a pinned requirements file and
passes it to the same `OsvScannerSpec.argv` the declared scan uses, so the two
halves of the verdict are comparable and one binary is trusted.

And the section that cannot be scanned is not folded into the number. Model
artifacts, vendored weights, MCP servers and skills get rows carrying
`advisory_source: none`, under the `coverage_gap` convention #688 shipped: an
inventoried-but-unvetted row is evidence the component is *known*, never that it
is clean.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.harness import supply_chain as sc
from app.paths import LIVE_CHECKOUT

#: The artifact the scan subcommand writes. Its own key beside the declared half
#: of the baseline (`observed:`), never merged into `advisory_count`, so an old
#: reader of `eval/supply-chain/baseline.yaml` sees exactly what it saw before.
OBSERVED_SCHEMA = "lloyd-supply-chain-observed/1"
OBSERVED_BASELINE_KEY = "observed"

#: The launcher/config files the venv scope is read from. Named because the
#: scope's whole claim is that it tracks what the box actually execs: a venv that
#: appears in none of them is not in the inventory, and that is a fact about the
#: scope rather than a silent omission.
SCOPE_FILES = (
    "agent-services/bin/start-qwen3-tts.sh",
    "agent-services/bin/start-djev.sh",
    "agent-services/bin/start-qwen38-flash-next.sh",
    "agent-services/bin/start-secondary.sh",
    "agent-services/bin/start-embedding.sh",
    "agent-services/supervisor/conf.d/agent-llm-primary.conf",
)
VENV_PREFIX = ".venvs/"
#: A user-site is a runtime target by the same argument as a venv: #688's own
#: exposure baseline recorded `pip install --user --break-system-packages
#: youtube-transcript-api` as the one real install event on this machine, and no
#: file the declared scan reads describes its destination.
USER_SITE_GLOB = ".local/lib/python3.*/site-packages"

#: The label of the one synthesised target that is not a venv: every observed
#: name+version pair on the box, deduplicated. The site label *is* the directory
#: name (`synthesized_lockfile_name`), so this row reads `all-sites/requirements.txt`.
MERGED_SITE = "all-sites"

#: A row's `advisory_source` for every component kind no advisory feed can see.
NO_ADVISORY_SOURCE = "none"
WEIGHT_ROOT = "agent-services/models"
SKILLS_REL = "skills"
#: Cap on a digest read, disclosed in the row as `bytes_hashed`. The 7 files
#: under `agent-services/models` total 7.3 MB and are hashed whole; a GGUF in a
#: HF cache is not read to make an inventory.
DIGEST_CAP = 4 * 1024 * 1024

#: osv-scanner's own count line, singular or plural. The `?` is load-bearing: a
#: 1-package target prints "found 1 package", and a plural-only pattern reports
#: no count for it, which a caller reading `None` as "zero parsed" would turn
#: into a clean verdict on a set the parser simply never described.
FOUND_PACKAGES_RE = re.compile(r"found (\d+) packages?\b")
#: `--lockfile <format>:<path>`: a format label, not part of the path. Stripping
#: it on the FIRST colon is correct here because a POSIX path carries no colon;
#: on Windows it would not be, and this box has no Windows path.
LOCKFILE_FORMAT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*:(?=/)")


@dataclass
class SiteRoot:
    """One installed-package root the box runs.

    `label` for `lloyd` is the venv name, so a report row reads as the engine it
    belongs to; the account home is a parameter, not `Path.home()`, because a
    gate round runs with `$HOME` redirected to the round's directory and the
    inventory must describe the machine or it describes nothing.
    """

    label: str
    path: Path
    interpreter: str          # python3.12 / python3.14, from the site dir itself
    origin: str               # which file named this venv, or "user-site"

    def as_row(self) -> dict[str, Any]:
        return {"label": self.label, "path": str(self.path),
                "interpreter": self.interpreter, "origin": self.origin,
                "exists": self.path.is_dir()}


@dataclass
class Component:
    """One installed distribution, in one root.

    `interpreter` is why this is not a set of names: the item's live gap is the
    3.14 user-site, and a row that loses which interpreter installed it cannot
    answer "is the thing with the disclosure on my box, and where".
    """

    name: str
    version: str
    site: str
    interpreter: str
    root: str

    def as_row(self) -> dict[str, Any]:
        return {"component": f"{self.name}=={self.version}@{self.site}",
                "name": self.name, "version": self.version, "site": self.site,
                "interpreter": self.interpreter, "site_packages": self.root,
                "component_kind": "python-distribution"}


def _stamp(moment: dt.datetime) -> str:
    """Offset-bearing UTC, the same rule as `supply_chain._utc_stamp`."""
    return moment.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def service_venv_names(repo_root: Path | str = LIVE_CHECKOUT,
                       *, scope_files: tuple[str, ...] = SCOPE_FILES) -> set[str]:
    """Venv names named by a launcher or a supervisor drop-in.

    Globs the bin directory rather than trusting `SCOPE_FILES`, so a launcher that
    exists and is not in the list is still read: the list is the *claim*, and a
    launcher found by the glob that is absent from the claim is reported by
    `scope_gaps` as a scope that needs widening — not silently added, because
    "which venvs run" is the one thing in this module that must be argued, not
    discovered by accident.
    """
    base = Path(repo_root)
    names: set[str] = set()
    for rel in scope_files:
        names |= _venvs_in(base / rel)
    return names


def _venvs_in(path: Path) -> set[str]:
    """`.venvs/<name>` paths named anywhere in one file's text."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    # One capture group, so findall yields the names themselves rather than spans
    # to split. No trailing separator is required: `start-djev.sh` names its venv
    # as `"$HOME/lloyd/.venvs/vllm-djev"` with the quote where a slash would be,
    # and a scope that missed that form would miss the engine it was written for.
    return {name for name in re.findall(r"\.venvs/([A-Za-z0-9._-]+)", text)
            if name and not name.startswith(".")}


def launcher_venv_names(repo_root: Path | str = LIVE_CHECKOUT) -> dict[str, str]:
    """`venv name -> first file that names it`, over the launchers and confs.

    The whole `agent-services` tree is read, not just the four named scripts,
    because the question "does a service exec this venv" is answered by any file
    in that tree and a stale allowlist cannot close a property that must hold over
    an open set.
    """
    base = Path(repo_root)
    origin: dict[str, str] = {}
    for path in sorted((base / "agent-services").rglob("*")):
        if not path.is_file() or path.suffix not in (".sh", ".conf", ".service"):
            continue
        rel = str(path.relative_to(base))
        for name in _venvs_in(path):
            origin.setdefault(name, rel)
    return origin


def site_roots(repo_root: Path | str = LIVE_CHECKOUT,
               account_home: Path | str | None = None) -> list[SiteRoot]:
    """Every installed-package root in scope, ordered by label."""
    home = Path(account_home) if account_home else Path.home()
    roots: list[SiteRoot] = []
    origin = launcher_venv_names(repo_root)
    for name in sorted(service_venv_names(repo_root)):
        for site in sorted(glob.glob(str(home / "lloyd" / VENV_PREFIX / name
                                         / "lib" / "python3*" / "site-packages"))):
            path = Path(site)
            # `path` is `.../lib/python3.12/site-packages`, so the interpreter tag
            # is the parent's name — the component that says which runtime these
            # packages belong to, and the one that distinguishes this venv from
            # the account's user-site three directories away.
            roots.append(SiteRoot(name, path, path.parent.name,
                                  origin.get(name, "named in SCOPE_FILES only")))
        # A venv named by a launcher with no site-packages under the account home
        # is a scope gap, not a root with zero components: see `scope_gaps`.
    for site in sorted(glob.glob(str(home / USER_SITE_GLOB))):
        path = Path(site)
        roots.append(SiteRoot("user-site", path, path.parent.name, "user-site"))
    return roots


def scope_gaps(repo_root: Path | str = LIVE_CHECKOUT,
               account_home: Path | str | None = None) -> list[str]:
    """Named-by-a-launcher-but-absent venvs, and launchers the claim does not list.

    Two different silences, both reported: a venv a service execs that has no
    site-packages here (a broken launcher, or an inventory running on the wrong
    box) and a launcher naming a venv that `SCOPE_FILES` does not admit. The
    second is the one that would otherwise let the scope rot quietly — a new
    engine's venv appearing in a launcher and in no claim.
    """
    home = Path(account_home) if account_home else Path.home()
    named = service_venv_names(repo_root)
    gaps: list[str] = []
    for name in sorted(named):
        pattern = str(home / "lloyd" / VENV_PREFIX / name / "lib" / "python3*"
                      / "site-packages")
        if not glob.glob(pattern):
            gaps.append(f"{name}: named by a launcher, no site-packages under "
                        f"{pattern}")
    claimed = set(SCOPE_FILES)
    origin = launcher_venv_names(repo_root)
    # venv -> the file that names it, inverted to file -> the venvs it names: the
    # gap reads "this launcher admits a venv the claim does not", so the message
    # has to name the venv the *file* brought in, not the file the venv came from.
    by_file: dict[str, list[str]] = {}
    for venv_name, rel in origin.items():
        by_file.setdefault(rel, []).append(venv_name)
    for rel in sorted(f for f, venvs in by_file.items()
                      if f not in claimed and venvs):
        gaps.append(f"{rel}: names venv(s) {sorted(by_file[rel])} from a file "
                    f"`scope` does not list, so the claim is narrower than the "
                    "launchers — either the launcher belongs in SCOPE_FILES or "
                    "its venv belongs to nothing that runs")
    return gaps


def installed_distributions(site_packages: Path | str) -> list[tuple[str, str]]:
    """`(name, version)` from every `*/dist-info/METADATA` under one root.

    Reads METADATA rather than running `pip list`: a pip invocation for five
    interpreters needs five interpreters to be healthy at scan time, and the
    answer to "what is installed" is already on disk as the file pip wrote. The
    pair `(name, version)` is what a scan target needs; `inventory` adds which
    root and interpreter it came from.

    Returns [] for an absent root, and `site_roots` is what records an absent root
    as a gap — a zero here is always paired with a gap entry there.
    """
    root = Path(site_packages)
    out: list[tuple[str, str]] = []
    if not root.is_dir():
        return out
    for meta in sorted(root.glob("*.dist-info/METADATA")):
        name = version = None
        try:
            text = meta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("Name:") and name is None:
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Version:") and version is None:
                version = line.split(":", 1)[1].strip()
            if name and version:
                break
        if name and version:
            out.append((name, version))
    return out


def inventory(roots: list[SiteRoot]) -> list[Component]:
    """One row per installed distribution per root — duplicates across venvs kept."""
    rows: list[Component] = []
    for root in roots:
        for name, version in installed_distributions(root.path):
            rows.append(Component(name, version, root.label, root.interpreter,
                                  str(root.path)))
    return rows


def observed_dir(account_home: Path | str = Path.home()) -> Path:
    """Where the synthesised scan targets live: the runtime root, never the checkout.

    A generated requirements file inside the repo would be picked up by the next
    `dependency_files()` scan and become a fourth *declared* lockfile — the exact
    category error this module is about, a file that describes what is installed
    read as a file that declares what is wanted. Same `app.paths.DATA_ROOT`
    convention as `registry_cache_path`, with the account fallback.
    """
    try:
        from app.paths import DATA_ROOT
        return Path(DATA_ROOT) / "supply-chain" / "observed"
    except Exception:  # noqa: BLE001 — a checker runs without app.paths too
        return Path(account_home) / "lloyd-data" / "supply-chain" / "observed"


def synthesize_lockfiles(rows: list[Component], out_dir: Path | str) -> dict[str, Path]:
    """One pinned `name==version` requirements file per site, written under `out_dir`.

    Per site rather than one merged file, because the answer to "where is it" is
    per site: `pillow==12.1.1` in the TTS venv and `pillow` absent from the main
    venv are one package and two exposures, and the merged count cannot say which
    engine to stop using. `out_dir` is under the runtime root, never the checkout:
    it is generated state, and a synthesised requirements file in the repo would
    become a fourth declared lockfile by the next `dependency_files()` glob.
    """
    out = Path(out_dir)
    by_site: dict[str, list[Component]] = {}
    for row in rows:
        by_site.setdefault(row.site, []).append(row)
    written: dict[str, Path] = {}
    for site, site_rows in sorted(by_site.items()):
        target_dir = out / site
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "requirements.txt"
        pairs = sorted({f"{r.name}=={r.version}" for r in site_rows})
        path.write_text("# synthesised from installed *.dist-info by "
                        "app.harness.supply_chain_inventory — do not edit, do not "
                        "declare, it is a scan target and nothing else\n"
                        + "\n".join(pairs) + "\n", encoding="utf-8")
        written[site] = path
    return written


def scanned_package_count(stderr: str) -> int | None:
    """What osv-scanner says it parsed, from its own count line.

    `None` when the line is absent. A caller must not read that as zero: an empty
    parse and an unreported parse are different facts, and clause 3 exists because
    a synthesised target the parser choked on has to fail loudly rather than
    arrive as `advisory_count: 0`.
    """
    found = FOUND_PACKAGES_RE.findall(stderr)
    return sum(int(n) for n in found) if found else None


def scan_site(site: str, lockfile: Path, spec, *, repo_root: Path,
              expected_count: int, timeout: float = 300.0) -> dict[str, Any]:
    """Scan one synthesised target through the shipped offline argv.

    Runs with `cwd=repo_root` because osv-scanner's offline prerequisite check is
    `database_dir.is_dir()` against a path the spec derives, and the argv the
    declared scan builds is the argv used here — a second, better target build
    would be a second definition of "scanned", which is what this whole item is
    about.
    """
    argv = list(spec.argv([lockfile]))
    base: dict[str, Any] = {"site": site, "lockfile": str(lockfile),
                            "argv": argv}
    try:
        proc = subprocess.run([str(spec.binary), *argv], capture_output=True,
                              text=True, check=False, env=dict(os.environ, **spec.env()),
                              cwd=str(repo_root), timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {**base, "scan_status": "failed", "packages_scanned": None,
                "advisories": None, "reason": f"{site}: {exc}"}
    payload = sc._parse_scan_output(spec.name, proc.stdout)
    parsed = scanned_package_count(proc.stderr)
    if payload is None:
        return {**base, "scan_status": "failed", "packages_scanned": parsed,
                "advisories": None,
                "reason": (f"{site}: {spec.name} exited {proc.returncode} and its "
                           f"output is not a report: {(proc.stderr or '')[-200:]}")}
    advisories, count = payload
    return {**base, "scan_status": "completed", "packages_scanned": parsed,
            "advisory_count": count,
            "advisories": [{"id": a.id, "package": a.package, "version": a.version}
                           for a in advisories],
            "reason": ""}


def database_freshness_for(spec: Any) -> str | None:
    """Which offline OSV database `spec`'s numbers came from, or None for none.

    #3544. A dated `inventory-<date>.json` used to carry `generated_at` and nothing
    else about its inputs, so two dated reports were not comparable without going and
    stat-ing `osv-db/*/all.zip`: on 2026-10-29 the headline went 150 -> 358 with the
    observed set byte-identical (906 components, zero added, zero removed) because the
    mirror refreshed on 2026-10-26 — pre-existing installations the frozen database
    was blind to, not new exposure. The stamp is what lets a reader tell a refresh
    from a regression from inside the artifact.

    `None` where no database was read: no scanner resolved, or the mirrored root holds
    no `all.zip`. `OsvScannerSpec.freshness` is the single source — the newest
    `all.zip` mtime under the mirrored db root — and is not recomputed here, so the
    report cannot claim an age the scanner's own prerequisite did not use. A spec
    without the method (`PipAuditSpec` reads a loopback OSV API, not this zip) reports
    no database rather than borrowing one it did not consult.
    """
    if spec is None:
        return None
    stamp = getattr(spec, "freshness", None)
    return stamp() if callable(stamp) else None


def observed_scan(repo_root: Path | str = LIVE_CHECKOUT,
                  out_dir: Path | str | None = None,
                  *, scanner_name: str | None = None,
                  account_home: Path | str | None = None,
                  components: list[dict[str, Any]] | None = None,
                  rows: list[Component] | None = None,
                  roots: list[SiteRoot] | None = None,
                  now: dt.datetime | None = None) -> dict[str, Any]:
    """Inventory, synthesise, scan, report — one call, one JSON-safe dict.

    Everything inspectable is injectable (`roots`, `rows`, `components`) so a test
    can plant a `.dist-info` in a temp root and watch it arrive in the report and
    in the scan target, which is the only way to prove the reader reads.

    `advisories_in_observed_set_that_declared_lockfiles_missed` is the item's
    headline number: advisories against a pinned version in the observed set whose
    *name* the declared lockfiles do not carry at all. Name-matching, not version
    — an observed `pip 25.0.1` against a declared `pip==25.0` is a declared package
    with a version the lockfile got wrong, and counting it as missed would report a
    bookkeeping error as a new exposure. The consequence is stated rather than
    hidden: an already-declared package installed at a *different* vulnerable
    version is NOT in this number, and `sites[].advisory_count` is where a
    per-site total is visible.
    """
    base = Path(repo_root)
    home = Path(account_home) if account_home else Path.home()
    moment = now or dt.datetime.now(dt.timezone.utc)
    root_list = roots if roots is not None else site_roots(base, home)
    row_list = rows if rows is not None else inventory(root_list)
    out = Path(out_dir) if out_dir else observed_dir(home)
    declared = sc.read_dependency_set(base)
    written = synthesize_lockfiles(row_list, out)

    spec, why = sc.resolve_scanner(scanner_name or "osv-scanner")
    expected = _rows_per_site(row_list)
    sites: list[dict[str, Any]] = []
    for site, lockfile in sorted(written.items()):
        if spec:
            one = scan_site(site, lockfile, spec, repo_root=base,
                            expected_count=expected.get(site, 0))
        else:
            one = {"site": site, "lockfile": str(lockfile), "argv": [],
                   "scan_status": "scanner_absent", "packages_scanned": None,
                   "advisories": None, "reason": why,
                   "components_declared_in_target": expected.get(site, 0),
                   "parse_count_matches_inventory": False}
        advisories = one.get("advisories")
        if advisories is None:
            one["advisories_total"] = None
            one["missed_advisories"] = None
            one["missed_packages"] = []
        else:
            missed = [a for a in advisories
                      if sc.normalize_dist_name(a["package"]) not in declared]
            one["advisories_total"] = len(advisories)
            one["missed_advisories"] = len(missed)
            one["missed_packages"] = sorted({a["package"] for a in missed})
        one.setdefault("components_declared_in_target", expected.get(site, 0))
        one.setdefault("parse_count_matches_inventory",
                       one["packages_scanned"] == expected.get(site, 0))
        one.pop("advisories", None)      # counts and names travel, 182 rows do not
        sites.append(one)

    # The same set asked a second way. Per-site counting is per installed copy —
    # one package in three venvs is three environments to patch — while a reader
    # working a package list wants distinct name+version packages. Both numbers
    # travel, because the reader who needs one does not want the other, and an
    # artifact that prints only one silently answers a question nobody asked.
    if spec and row_list:
        merged_dir = out / MERGED_SITE
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_file = merged_dir / "requirements.txt"
        merged_rows = sorted({(r.name, r.version) for r in row_list})
        merged_file.write_text(
            "\n".join(f"{n}=={v}" for n, v in merged_rows) + "\n", encoding="utf-8")
        merged = scan_site(MERGED_SITE, merged_file, spec, repo_root=repo_root,
                           expected_count=len(merged_rows))
        merged["lockfile"] = str(merged_file)
        merged["advisories_in_observed_set"] = None
        merged["advisories_in_observed_set_that_declared_lockfiles_missed"] = None
        merged["distinct_missed_packages"] = None
        merged["distinct_missed_package_names"] = None
        if merged["scan_status"] == "completed":
            mm = [a for a in merged["advisories"]
                  if sc.normalize_dist_name(a["package"]) not in declared]
            merged["advisories_in_observed_set"] = len(merged["advisories"])
            merged["advisories_in_observed_set_that_declared_lockfiles_missed"] = len(mm)
            merged["distinct_missed_packages"] = len({(a["package"], a["version"])
                                                      for a in mm})
            merged["distinct_missed_package_names"] = sorted({a["package"] for a in mm})
        merged.pop("advisories", None)
    else:
        merged = {"site": MERGED_SITE, "scan_status": "scanner_absent",
                  "packages_scanned": None, "advisories_in_observed_set": None,
                  "advisories_in_observed_set_that_declared_lockfiles_missed": None,
                  "distinct_missed_packages": None,
                  "distinct_missed_package_names": None,
                  "lockfile": None, "reason": "no scanner, so no merged target ran"}

    gap_sites = [s["site"] for s in sites if s["scan_status"] != "completed"]
    merged_gap = merged["scan_status"] != "completed"
    complete = bool(spec) and bool(sites) and not gap_sites and not merged_gap
    mismatched = [s["site"] for s in sites
                  if s["scan_status"] == "completed"
                  and not s["parse_count_matches_inventory"]]

    def _sum(field: str) -> int | None:
        return None if gap_sites else sum(int(s.get(field) or 0) for s in sites)

    non_advisory, non_advisory_gaps = (
        (components, []) if components is not None
        else non_advisory_rows(base, home))

    return {
        "schema": OBSERVED_SCHEMA,
        "generated_at": _stamp(moment),
        "scope": {
            "derived_from": list(SCOPE_FILES),
            "venv_names": sorted(service_venv_names(base)),
            "user_site_glob": USER_SITE_GLOB,
            "roots": [r.as_row() for r in root_list],
            "gaps": scope_gaps(base, home),
            "decision": "every site-packages a service launcher or the supervisor "
                        "conf execs, plus every ~/.local/lib/python3.*/site-packages "
                        "— the importable set of the processes that run, not a disk "
                        "census",
        },
        "scanner": {"name": scanner_name or "osv-scanner",
                    "resolved_path": str(spec.binary) if spec else None,
                    "resolved": spec is not None, "reason": why},
        # Present always, null when no database was read (#3544). Scalar and beside
        # `scanner` on purpose: `scanner` says what ran, this says what it read, and a
        # reader comparing two dated reports compares these two lines before they
        # compare any advisory number.
        "database_freshness": database_freshness_for(spec),
        "component_count": len(row_list),
        "distinct_component_versions": len({(r.name, r.version) for r in row_list}),
        "components": [r.as_row() for r in
                       sorted(row_list, key=lambda r: (r.site, r.name, r.version))],
        "sites": sites,
        "merged_scan": merged,
        "scan_status": "completed" if complete else "coverage-gap",
        "sites_not_scanned": gap_sites,
        "target_parse_mismatches": mismatched,
        "advisories_in_observed_set_total": _sum("advisories_total"),
        "advisories_in_observed_set_that_declared_lockfiles_missed":
            _sum("missed_advisories"),
        "non_advisory_components": non_advisory,
        "non_advisory_component_count": len(non_advisory),
        "non_advisory_gaps": non_advisory_gaps,
        "non_advisory_advisory_count": advisory_count_from(non_advisory),
        "reason": ("" if complete else
                   f"{'no offline scanner: ' + why if not spec else 'a site failed: ' + ', '.join(gap_sites)}. "
                   "No observed advisory count is reported for an unscanned set: "
                   "a coverage gap is not a zero"),
    }


def _rows_per_site(rows: list[Component]) -> dict[str, int]:
    return {site: len({(r.name, r.version) for r in rows if r.site == site})
            for site in {r.site for r in rows}}


# ── the component kinds no advisory feed can see ────────────────────────────
#
# Garza's framing for why this half exists: a lockfile is an SBOM for one
# language of a system that has several component kinds, and the disclosure that
# reaches a model artifact or a vendored weight resolves to nothing in a
# requirements file. #688's own triage reached the same place from the other side
# — "a model artifact pulled from Hugging Face ... is a supply-chain surface OSV
# will never see". The answer here is not a scanner for them (no offline one
# exists); it is a row that can be queried and that can never be counted clean.

def sha256_of(path: Path, *, limit: int = DIGEST_CAP) -> str:
    """Content digest, first `limit` bytes, empty string if the file is unreadable.

    The cap is disclosed in the row as `bytes_hashed` rather than assumed: a 30 GiB
    GGUF is not read to make an inventory, and a truncated digest that does not say
    so is worse than none because it reads as a fingerprint. For the 7 files under
    `agent-services/models` (7.3 MB, largest 2,327,524 bytes) the cap never binds.
    """
    digest = hashlib.sha256()
    hashed = 0
    try:
        with path.open("rb") as handle:
            while hashed < limit:
                chunk = handle.read(min(1 << 20, limit - hashed))
                if not chunk:
                    break
                digest.update(chunk)
                hashed += len(chunk)
    except OSError:
        return ""
    return digest.hexdigest()[:16] if hashed else ""


def weight_rows(repo_root: Path | str = LIVE_CHECKOUT) -> list[dict[str, Any]]:
    """Vendored weights in the checkout, by path, size and digest.

    `version` is null where there is no version to record: an ONNX file carries no
    metadata naming the upstream release it came from, and inventing a string for a
    field that is supposed to be exact would be the same fabrication in the other
    direction. `agent-services/models/silero-vad/SOURCE` survives as a `provenance`
    note on the row for the weight beside it, which is the honest form of "we know
    where this came from, and it is not a version".
    """
    base = Path(repo_root)
    root = base / WEIGHT_ROOT
    if not root.is_dir():
        return []
    notes: dict[str, str] = {}
    for note in sorted(root.rglob("SOURCE")):
        try:
            notes[note.parent.name] = note.read_text(
                encoding="utf-8", errors="replace").strip()[:300]
        except OSError:
            continue
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SOURCE"):
        size = path.stat().st_size
        row: dict[str, Any] = {
            "component": f"weight:{path.relative_to(base)}",
            "component_kind": "vendored-weight",
            "name": path.name, "version": None, "site": WEIGHT_ROOT,
            "interpreter": None, "path": str(path.relative_to(base)),
            "size_bytes": size, "sha256": sha256_of(path),
            "bytes_hashed": "whole-file" if size <= DIGEST_CAP else "first-4MiB",
            "advisory_source": NO_ADVISORY_SOURCE,
        }
        if path.parent.name in notes:
            row["provenance"] = notes[path.parent.name]
        rows.append(row)
    return rows


def mcp_server_rows(repo_root: Path | str = LIVE_CHECKOUT) -> list[dict[str, Any]]:
    """MCP servers declared in `config.yaml`.

    Parsed with a minimal indent-scoped pass rather than the harness's config
    loader: this row is inventory, not a config decision, and reading `config.yaml`
    through a loader that applies defaults would put a *derived* value in an
    inventory that is supposed to say what the file says. `version` is null — a
    server here is this checkout's own module, so its version is the tree commit
    the declared half records in `tree.commit`.
    """
    path = Path(repo_root) / "config.yaml"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    current: dict[str, str] | None = None
    in_block = False
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0:
            in_block = stripped == "mcp_servers:"
            current = None
            continue
        if not in_block:
            continue
        if indent == 2 and stripped.endswith(":"):
            current = {"name": stripped[:-1].strip(), "url": ""}
            rows.append({"component": f"mcp:{current['name']}",
                         "component_kind": "mcp-server", "name": current["name"],
                         "version": None, "site": "config.yaml",
                         "interpreter": None, "url": "",
                         "advisory_source": NO_ADVISORY_SOURCE})
        elif indent == 4 and current is not None and stripped.startswith("url:"):
            url = stripped.partition(":")[2].strip().strip("'\"")
            current["url"] = url
            rows[-1]["url"] = url
    return [r for r in rows if r["url"]]      # url-less blocks are commented stubs


def skill_rows(vault_root: Path | str) -> list[dict[str, Any]]:
    """One row per skill the loader would see: `<skills>/<name>/SKILL.md`.

    Skills live in the vault, not the checkout (`architecture/skills.md`: the
    library moved to `~/obsidian/skills/`), and the root is passed in by the
    caller rather than reached for, so a test can plant a skill and a wrong box
    cannot pass by reading the real one. `version` is null for most of them, which
    is the gap this section exists to make visible rather than paper over.
    """
    root = Path(vault_root) / SKILLS_REL
    rows: list[dict[str, Any]] = []
    for skill_md in sorted(root.glob("*/SKILL.md")) if root.is_dir() else []:
        name = skill_md.parent.name
        version = None
        try:
            head = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        in_fm = False
        for line in head.splitlines():
            if line.strip() == "---":
                if in_fm:
                    break
                in_fm = True
                continue
            if in_fm and line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip("'\"") or name
            elif in_fm and line.startswith("version:"):
                version = line.split(":", 1)[1].strip().strip("'\"") or None
        size = skill_md.stat().st_size
        rows.append({"component": f"skill:{name}", "component_kind": "skill",
                     "name": name, "version": version, "site": SKILLS_REL,
                     "interpreter": None,
                     "path": str(skill_md.relative_to(Path(vault_root))),
                     "size_bytes": size, "sha256": sha256_of(skill_md),
                     "bytes_hashed": "whole-file" if size <= DIGEST_CAP else "first-4MiB",
                     "advisory_source": NO_ADVISORY_SOURCE})
    return rows


def model_rows(account_home: Path | str = Path.home()) -> list[dict[str, Any]]:
    """Downloaded model artifacts, by Hugging Face cache snapshot.

    A local artifact has no version string that means upstream — the engines load
    a snapshot directory, so `version` is null and identity is the blob set plus
    digests, which is what a disclosure has to resolve against. Returns [] with a
    gap entry when there is no cache, and the gap is what distinguishes "this box
    loads no HF model" from "this module looked in the wrong place".
    """
    home = Path(account_home)
    rows: list[dict[str, Any]] = []
    for hub in sorted(home.glob("huggingface/hub/models--*")):
        for snap in sorted((hub / "snapshots").glob("*")) if (hub / "snapshots").is_dir() else []:
            files = sorted(p for p in snap.rglob("*") if p.is_file())
            if not files:
                continue
            digest = hashlib.sha256()
            total = 0
            for f in files:
                digest.update(f.name.encode())
                total += f.stat().st_size
            rows.append({"component": f"model:{hub.name}/{snap.name[:12]}",
                         "component_kind": "model-artifact",
                         "name": hub.name.removeprefix("models--").replace("--", "/"),
                         "version": None, "site": str(snap.parent.parent),
                         "interpreter": None, "revision": snap.name,
                         "file_count": len(files), "size_bytes": total,
                         "sha256": digest.hexdigest()[:16],
                         "bytes_hashed": "filename-set",
                         "advisory_source": NO_ADVISORY_SOURCE})
    return rows


def non_advisory_rows(repo_root: Path | str = LIVE_CHECKOUT,
                      account_home: Path | str | None = None,
                      vault_root: Path | str | None = None
                      ) -> tuple[list[dict[str, Any]], list[str]]:
    """Every component kind no advisory feed can see, plus the gaps.

    Derives the vault root the way the loader does, falling back to the account
    convention and naming which it used — the moved-root class from the
    `~/lloyd-data` incident, where a stored path that names a root which has moved
    reads as a clean empty result.
    """
    home = Path(account_home) if account_home else Path.home()
    vault = Path(vault_root) if vault_root else _vault_root()
    rows = weight_rows(repo_root) + mcp_server_rows(repo_root) + skill_rows(vault)
    gaps: list[str] = []
    if not rows:
        gaps.append("no weights, MCP servers or skills found — a clean empty "
                    "section, not a scanned one")
    vault_skills = sum(1 for r in rows if r["component_kind"] == "skill")
    if not vault_skills:
        gaps.append(f"no skills under {vault / SKILLS_REL}")
    models = model_rows(home)
    if not models:
        gaps.append(f"no Hugging Face cache snapshots under "
                    f"{home / 'huggingface/hub'} — this box's model artifacts are "
                    "the vendored weights above plus whatever an engine loads from "
                    "a path no inventory names")
    rows += models
    return rows, gaps


def _vault_root() -> Path:
    """The vault, derived the way `app.paths` derives it — never hard-coded."""
    try:
        from app.paths import VAULT_ROOT
        return Path(VAULT_ROOT)
    except Exception:  # noqa: BLE001 — the checker runs without app.paths too
        return Path.home() / "obsidian"


def advisory_count_from(rows: list[dict[str, Any]]) -> int:
    """Advisories implied by component rows: this section is structurally zero.

    Not a helper to call — a rule with a name and a test. The mechanism that makes
    "adding a non-advisory row cannot move the verdict" true is that the count is
    defined over rows that *have* an advisory source, and a docstring hoping a
    reader infers that is how a count drifts.
    """
    return sum(int(r.get("advisory_count") or 0) for r in rows
               if r.get("advisory_source") != NO_ADVISORY_SOURCE)


def attach_to_baseline(baseline: dict[str, Any], observed: dict[str, Any],
                       ) -> dict[str, Any]:
    """The committed artifact: declared half, plus `observed:` beside it.

    The declared keys are copied byte-for-byte so every existing reader of
    `eval/supply-chain/baseline.yaml` sees what it saw before and the counts it
    reads are still the counts of the scan that made them; the observed half lives
    under its own key with its own `scan_status`. Nothing reconciles the two
    advisory numbers into one, deliberately: they answer different questions, and
    a single number is how a coverage gap gets laundered into a clean verdict.
    """
    merged = dict(baseline)
    merged[OBSERVED_BASELINE_KEY] = {k: v for k, v in observed.items()
                                     if k not in ("schema", "generated_at")}
    merged["observed_schema"] = observed.get("schema", OBSERVED_SCHEMA)
    return merged


def write_merged_baseline(report, observed: dict[str, Any],
                          path: Path | str | None = None, *,
                          root: Path | str | None = None) -> Path:
    """Commit the declared half and the observed half in one artifact.

    Goes through `supply_chain.write_baseline` rather than writing YAML itself, so
    the target guard runs on the same paths the declared scan named and a baseline
    describing a deleted worktree is refused here too — the guard is worth nothing
    if the richer writer bypasses it. The declared keys are then re-emitted
    unchanged with `observed:` appended beside them.

    `path`/`root` are the shipped writer's own escape hatch: a regeneration aimed
    at the production checkout writes into the tree a person can open, and the
    guard checks the scan's targets against that checkout either way.
    """
    import yaml

    target = sc.write_baseline(report, path, root=root)
    text = target.read_text(encoding="utf-8")
    header = "".join(line + "\n" for line in text.splitlines()
                     if line.startswith("#"))
    declared = yaml.safe_load(text[text.index("schema:"):])
    merged = attach_to_baseline(declared, observed)
    merged["observed_generated_at"] = observed.get("generated_at")
    body = yaml.safe_dump(merged, sort_keys=False, width=100)
    target.write_text(header + body, encoding="utf-8")
    return target


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="supply_chain_inventory",
                                     description="Observed component inventory")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("inventory", help="enumerate, synthesise and scan")
    scan.add_argument("--json", action="store_true")
    scan.add_argument("--scanner", default=None)
    scan.add_argument("--out", default=None)
    scan.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    report = observed_scan(Path(args.root) if args.root else LIVE_CHECKOUT,
                           Path(args.out) if args.out else None,
                           scanner_name=args.scanner)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"observed inventory ({report['schema']}) "
              f"{report['component_count']} components, "
              f"{report['distinct_component_versions']} distinct versions")
        for root in report["scope"]["roots"]:
            print(f"  root {root['label']:22s} {root['interpreter']:11s} {root['path']}")
        for gap in report["scope"]["gaps"]:
            print(f"  scope gap: {gap}")
        for site in report["sites"]:
            print(f"  site {site['site']:22s} {site['scan_status']:14s} "
                  f"parsed={site['packages_scanned']!r} "
                  f"of {site['components_declared_in_target']} installed, "
                  f"missed={site['missed_advisories']!r}")
        print(f"  non-advisory components: {report['non_advisory_component_count']}")
        print(f"  advisories observed-but-not-declared: "
              f"{report['advisories_in_observed_set_that_declared_lockfiles_missed']!r}")
        if report["scan_status"] != "completed":
            print(f"  coverage gap: {report['reason']}")
    return 0 if report["scan_status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
