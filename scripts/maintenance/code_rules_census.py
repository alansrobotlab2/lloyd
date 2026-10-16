#!/usr/bin/env python3
"""Whole-tree census of the content rules (#2810) — the weekly half of a diff-scoped rung.

`scripts/automod/code_rules.py` runs the shipped rules over a round's *diff*. That is
the right unit for refusing a change, and it has a blind spot with a long memory: a
shape that predates every round is invisible to it, so a rule scoped to 354 of the
tree's files reports zero and reads like a clean tree. `run()` already takes
`files=` with `whole_tree=True` — #2574 built it and nothing outside the tests ever
called it. This is that caller: every tracked `.py`, once a week, on a schedule.

WHAT THE ROW CARRIES. Per rule, its own coverage (`covered` of `of`) beside its hits,
plus the tree totals: files scanned, allowlisted sites, suppression markers counted,
parse errors. Coverage sits next to the hit count on purpose. A rule's zero means
"nothing of this shape among the 411 files my scope read", and the only number that
distinguishes that from a scope that quietly stopped matching anything is the
denominator, which is exactly what a reader of a month-old row cannot otherwise
reconstruct.

WHAT A HIT DOES. It files ONE backlog item naming the rule and each site, and the
process still exits 0. The rung is observe-only by #2574's own split, and a weekly
pass that failed a round for a shape that was already on the tree on the day the rule
was written would be a way of making the oldest code unmergeable. A hit is a finding
for a person or a future round; it is not a red light. Filing is also deduplicated by
signature: an earlier census row that already filed this same set of sites files
nothing new, so a long-lived site cannot generate one item per week.

WHEN IT DOES EXIT NON-ZERO. Only when the census itself could not run — a missing rule
file, a tree `git` will not read. That is `unevaluated`, the runner's own verdict for
a read it could not do, and it is the one state where printing zero hits would be a
lie: #679 was closed on exactly that, "a green nobody can interrogate is worse than no
green". Exit 2 therefore means "the instrument is broken", never "the tree is dirty".

ONE ROW PER UTC DAY, AND THE RUN THAT CANNOT WRITE STILL REPORTS. A row exists for
today's date, this run writes nothing and files nothing: the day's first measurement
is the one the next week diffs against, and letting a late re-run replace it is the
#1576 failure (three rows 39 s apart, the following day diffed against the wrong one).
The report it prints is the fresh one, so a re-run is informative without being
durable. Nothing here writes inside the git tree: rows live under the data home
(`_pipeline/code-rules-census/`), and the only git command run is `ls-files`.

Driven weekly. The item offered a repo timer as the alternative and triage rejected
it: `scripts/maintenance/check-unit-enabledness.sh` enumerates every timer in the
repo and prints a `FAIL <unit> is not enabled` row for any landed-but-disabled one,
and enabling is host state no diff can produce — so a new `.timer` would be a
permanent red row. The drive is an autonomy task with `frequency: weekly` naming this
command (precedents: `autonomy/90-corpus-shape-trend.md`,
`autonomy/94-referential-integrity-ledger.md`), which is a vault change and is owed
after this lands, not a path a code round can add.

    python -m scripts.maintenance.code_rules_census [--tree ~/lloyd] [--out DIR]
        [--rules-path FILE] [--no-file] [--quiet]

Exit 0: the census ran, hits filed if there were any. Exit 2: it could not run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.automod import code_rules  # noqa: E402  (stdlib-only itself)

SCHEMA = 1
FILE_PREFIX = "code-rules-census-"
#: The board a finding lands on. The census audits this repository's code, which is
#: Lloyd's own board — the same one the item that produced it sits on.
BOARD = "lloyd"
TAGS = ("spawned-by-autocode", "code-rules")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def tracked_python(tree: Path) -> list[str]:
    """Every tracked `.py` in `tree` — the census's whole denominator."""
    proc = subprocess.run(["git", "-C", str(tree), "ls-files", "*.py"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`git ls-files *.py` in {tree} exited "
                           f"{proc.returncode}: {proc.stderr.strip()[:200]}")
    return [p for p in proc.stdout.splitlines() if p]


def hit_signature(hits: list[dict]) -> str:
    """A stable id for *this set* of findings, so a re-run files nothing new.

    Rule and path with multiplicity, deliberately not the line number: an unrelated
    edit that shifts a file by three lines is the same finding, and a signature that
    moved with it would file a fresh item for a site nobody touched.
    """
    key = "\n".join(sorted(f"{h['rule']}:{h['path']}" for h in hits))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def census(tree: Path, rules_path: Path | None = None) -> dict:
    """Run every rule over every tracked `.py`, and fold it into one row."""
    try:
        files = tracked_python(tree)
    except RuntimeError as exc:
        return {"status": code_rules.UNEVALUATED, "reason": str(exc), "rules": []}

    res = code_rules.run("HEAD", tree, files=files, rules_path=rules_path,
                         whole_tree=True)
    if res.get("status") != code_rules.EVALUATED:
        return {"status": code_rules.UNEVALUATED,
                "reason": res.get("reason", "the runner returned no status"),
                "rules": []}

    totals = res["totals"]
    coverage: dict[str, dict[str, int]] = totals["coverage"]
    counts = res["counts"]
    allowed_by_rule: dict[str, int] = {}
    for a in res["allowed"]:
        allowed_by_rule[a["rule"]] = allowed_by_rule.get(a["rule"], 0) + 1

    now = utc_now()
    row: dict = {
        "schema": SCHEMA,
        # One reading of the clock for both fields, or a run that crosses midnight
        # writes a row named for yesterday carrying a timestamp from today.
        "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": now.strftime("%Y-%m-%d"),
        "status": code_rules.EVALUATED,
        "scope": {"kind": "whole_tree", "tracked_py": sum(1 for f in files
                                                          if f.endswith(".py"))},
        "rules_loaded": res["rules_loaded"],
        "rules": [{"id": rid,
                   "covered": coverage[rid]["covered"],
                   "of": coverage[rid]["of"],
                   "hits": counts.get(rid, 0),
                   "allowed": allowed_by_rule.get(rid, 0)}
                  for rid in sorted(coverage)],
        "totals": {"files_scanned": totals["files"],
                   "hits": len(res["hits"]),
                   "allowed_sites": len(res["allowed"]),
                   "suppression_markers": totals["suppression_markers"],
                   "parse_errors": totals["parse_errors"]},
        "hits": [{"rule": h["rule"], "path": h["path"], "line": h["line"]}
                 for h in res["hits"]],
    }
    row["signature"] = hit_signature(row["hits"])
    return row


# ---------------------------------------------------------------------------
# The durable series: one row per UTC date, written with O_EXCL, never rewritten.
# ---------------------------------------------------------------------------

def default_out_dir() -> Path:
    """The data home's `_pipeline`, same precedence as `corpus_shape._pipeline_default`.

    Re-spelled rather than imported because a round's worktree is a git tree too: the
    anchored fallback has to be able to say "this tree holds nothing, use the machine's
    data root", and a census row written into a worktree is a row nothing reads.
    """
    data = os.environ.get("LLOYD_DATA")
    if data:
        return Path(data).expanduser() / "_pipeline" / "code-rules-census"
    anchored = REPO_ROOT / ".lloyd-data" / "_pipeline" / "code-rules-census"
    return anchored if anchored.exists() else (Path.home() / "lloyd-data"
                                               / "_pipeline" / "code-rules-census")


def row_path(out_dir: Path, date: str) -> Path:
    return out_dir / f"{FILE_PREFIX}{date}.json"


def prior_rows(out_dir: Path, *, before: str) -> list[dict]:
    """Every readable row for a date strictly earlier than `before`, oldest first."""
    if not out_dir.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(out_dir.glob(FILE_PREFIX + "*.json")):
        date = p.name[len(FILE_PREFIX):-len(".json")]
        if len(date) != 10 or date >= before:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            data.setdefault("_row_date", date)
            out.append(data)
    return out


def write_row(out_dir: Path, row: dict) -> tuple[Path, bool]:
    """Create today's row. Returns `(path, written)`; a day that has one keeps it.

    O_EXCL, never an open-for-write: the row a later week diffs against must be the one
    this day measured first (#1576). A same-day re-run returns the existing path with
    `written=False` and touches nothing, which is also what lets a row be made
    read-only by hand without the next run silently undoing that.
    """
    path = row_path(out_dir, row["date"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return path, False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row, indent=1, sort_keys=True) + "\n")
    return path, True


# ---------------------------------------------------------------------------
# Filing: a hit becomes one item, and stays one item.
# ---------------------------------------------------------------------------

def _already_filed(prior: list[dict], signature: str) -> int | None:
    """The item an earlier row already filed for this same set of sites."""
    for row in reversed(prior):
        if row.get("signature") != signature:
            continue
        filed = (row.get("filing") or {}).get("item")
        if filed:
            return int(filed)
    return None


def file_hits(row: dict, *, backlog_dir: Path | None = None,
              prior: list[dict] | None = None) -> dict:
    """File at most one backlog item for `row`'s hits. Never raises.

    Lazily imports the backlog writer, because that module pulls a YAML library and
    the census's own half of this job must keep working on the stock interpreter —
    the census is a check, and a check that cannot import is a check that reports
    nothing. A failure to file is recorded in the row and the run still exits 0: the
    row is the durable record either way.
    """
    filing: dict = {"item": None, "hits": len(row.get("hits") or [])}
    if not row.get("hits"):
        filing["skipped"] = "no hits"
        return filing
    already = _already_filed(prior or [], row["signature"])
    if already:
        filing["skipped"] = f"already filed as #{already}"
        return filing
    try:
        from scripts.automod import backlog  # noqa: E402 — needs pyyaml, hence lazy
    except Exception as exc:                                # noqa: BLE001 — reported, not fatal
        filing["error"] = f"could not import the backlog writer: {exc}"
        return filing

    sites = sorted({f"{h['rule']} at {h['path']}:{h['line']}" for h in row["hits"]})
    body = (
        "The weekly whole-tree census (`scripts/maintenance/code_rules_census.py`) "
        f"found these sites outside any round's diff, which is why the diff-scoped "
        f"rung never reported them. Census row `{row['date']}`, "
        f"{len(row['hits'])} hit(s), signature `{row['signature']}`.\n\n"
        + "\n".join(f"- `{s}`" for s in sites[:40])
        + ("\n- …and more; see the row." if len(sites) > 40 else "")
        + "\n\nFixing a site means changing the code, not widening an allowlist: the "
        "rule's own file names the incident that produced it. If the site is "
        "legitimately an exception, that is a ruling for a person, and the allow "
        "entry has to carry its own why.\n")
    try:
        item = backlog.new_item(
            f"Content-rule census found {len(row['hits'])} site(s) the diff rung cannot see",
            body, tags=list(TAGS), board=BOARD, backlog_dir=backlog_dir)
        filing["item"] = int(item.id)
    except Exception as exc:                                # noqa: BLE001 — reported, not fatal
        filing["error"] = f"could not file: {exc}"
    return filing


# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    tree = Path(args.tree).expanduser().resolve()
    rules_path = Path(args.rules_path).expanduser() if args.rules_path else None
    out_dir = Path(args.out).expanduser() if args.out else default_out_dir()

    try:
        row = census(tree, rules_path)
    except Exception as exc:                                # noqa: BLE001 — a census that cannot run must say so
        print(f"census could not run: {exc}", file=sys.stderr)
        return 2

    if row["status"] != code_rules.EVALUATED:
        print(f"census unevaluated: {row['reason']}", file=sys.stderr)
        return 2

    prior = prior_rows(out_dir, before=row["date"])
    if args.no_file:
        filing: dict = {"item": None, "hits": len(row["hits"]), "skipped": "--no-file"}
    else:
        filing = file_hits(row, backlog_dir=(Path(args.backlog_dir).expanduser()
                                             if args.backlog_dir else None),
                           prior=prior)
    row["filing"] = filing
    path, written = write_row(out_dir, row)

    if not args.quiet:
        print(json.dumps(row, indent=1, sort_keys=True) if args.json else _report(
            row, path, written))
    return 0


def _report(row: dict, path: Path, written: bool) -> str:
    lines = [f"census {row['date']}: {row['totals']['hits']} hit(s) over "
             f"{row['totals']['files_scanned']} scanned file(s) of "
             f"{row['scope']['tracked_py']} tracked .py; "
             f"{row['totals']['allowed_sites']} allowlisted site(s), "
             f"{row['totals']['suppression_markers']} suppression marker(s), "
             f"{row['totals']['parse_errors']} parse error(s)"]
    for r in row["rules"]:
        lines.append(f"  {r['id']}: covered {r['covered']} of {r['of']}, "
                     f"{r['hits']} hit(s), {r['allowed']} allowlisted")
    for h in row["hits"][:40]:
        lines.append(f"  HIT {h['rule']} at {h['path']}:{h['line']}")
    filing = row.get("filing") or {}
    if filing.get("item"):
        lines.append(f"  filed backlog item #{filing['item']}")
    elif filing.get("skipped"):
        lines.append(f"  filing skipped: {filing['skipped']}")
    if filing.get("error"):
        lines.append(f"  filing FAILED: {filing['error']}")
    lines.append(f"  row {'written' if written else 'already existed'}: {path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", default=str(REPO_ROOT),
                    help="the git tree to census (default: this checkout)")
    ap.add_argument("--out", help="row directory (default: the data home's "
                                  "_pipeline/code-rules-census)")
    ap.add_argument("--rules-path", help="rule file (default: the shipped one)")
    ap.add_argument("--backlog-dir", help="backlog directory for a filed item "
                                          "(default: the live backlog)")
    ap.add_argument("--no-file", action="store_true",
                    help="report hits without filing anything")
    ap.add_argument("--json", action="store_true", help="print the row as JSON")
    ap.add_argument("--quiet", action="store_true", help="print nothing")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
