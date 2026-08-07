#!/usr/bin/env python3
"""
qmd index maintenance — nightly orphan prune + embedding backfill.

WHY THIS EXISTS (2026-08-06, backlog #380):
qmd never prunes its own index. Orphaned embedding chunks — rows in
`content_vectors` whose `hash` no longer exists in `content` — accumulate on
every re-index. Left alone for months they reached **2,948,805 of 2,962,138
rows (99.5%)**, a 24 GB index, and 700 ms of single-core CPU per vector query
(the vec search brute-force scans every vector and applies the collection
filter afterwards, so cost is O(total vectors) regardless of query scope).

It is not only a speed problem: orphans displace real results. A vec-only query
returned 4 hits before the first cleanup and 20 after, and eval multi-hop MRR
went 0.022 → 0.220.

Separately, `qmd status` can report documents that are indexed but never
embedded. 406 such documents had accumulated because embedding silently stalled
while the vec leg was crash-looping.

SAFETY CONTRACT — this runs unattended:
  * The daemon is stopped ONLY when there is real work to do.
  * The daemon restart is in a `finally` block. Leaving qmd down would take
    vault retrieval offline entirely, which is far worse than a bloated index.
  * Every subprocess has a timeout.
  * Exit code is non-zero only when the daemon is left unhealthy — a failed
    prune with a healthy daemon is a warning, not a page.

Usage:
  python scripts/maintenance/qmd_index_maintenance.py            # act if needed
  python scripts/maintenance/qmd_index_maintenance.py --dry-run  # report only
  python scripts/maintenance/qmd_index_maintenance.py --force    # prune anyway
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QMD_CLI = Path.home() / ".bun/install/global/node_modules/@tobilu/qmd/dist/cli/qmd.js"
INDEX = Path.home() / ".cache/qmd/index.sqlite"
SUPERVISORCTL = Path.home() / ".local/share/uv/tools/supervisor/bin/supervisorctl"
SUPERVISOR_CONF = Path.home() / "lloyd/agent-services/supervisor/supervisord.conf"
SERVICE = "agent-qmd-daemon"
REPORT_DIR = Path.home() / "lloyd/_pipeline/reflection"

# Prune when orphans exceed this share of all vectors. Cleanup takes an
# exclusive lock and the daemon must be down for it, so don't pay that for a
# handful of rows. At the observed accumulation rate this trips every few days.
ORPHAN_RATIO_TRIGGER = 0.20
ORPHAN_ABS_TRIGGER = 50_000

QMD_ENV = {
    "HOME": str(Path.home()),
    "PATH": f"/opt/cuda/bin:{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_PATH": "/opt/cuda",
    "CUDA_HOME": "/opt/cuda",
    "LD_LIBRARY_PATH": "/usr/lib:/opt/cuda/lib64",
    "QMD_VEC_BACKEND": "bit",
}


def _sh(cmd: list[str], timeout: int, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return 1, repr(e)


def supervisor(action: str, timeout: int = 120) -> tuple[int, str]:
    return _sh([str(SUPERVISORCTL), "-c", str(SUPERVISOR_CONF), action, SERVICE], timeout)


def inspect_index() -> dict:
    """Read orphan counts straight from SQLite (read-only, daemon can be up)."""
    out: dict = {"index_bytes": INDEX.stat().st_size if INDEX.exists() else 0}
    if not INDEX.exists():
        out["error"] = "index missing"
        return out
    try:
        con = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True, timeout=30)
        q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
        out["vectors_total"] = q("select count(*) from content_vectors")
        out["vectors_orphaned"] = q(
            "select count(*) from content_vectors v "
            "left join content c on v.hash = c.hash where c.hash is null"
        )
        out["documents"] = q("select count(*) from documents")
        con.close()
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
        return out
    tot = out.get("vectors_total") or 0
    out["vectors_live"] = tot - out.get("vectors_orphaned", 0)
    out["orphan_ratio"] = round(out["vectors_orphaned"] / tot, 4) if tot else 0.0
    return out


def pending_embeddings() -> int:
    """Docs indexed but not embedded, per `qmd status`. -1 if unreadable."""
    rc, out = _sh(["/usr/bin/node", str(QMD_CLI), "status"], 180, env=QMD_ENV)
    if rc != 0:
        return -1
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Pending:"):
            digits = "".join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else 0
    return 0


def daemon_healthy(retries: int = 10) -> bool:
    """True once the daemon answers a trivial query."""
    for _ in range(retries):
        rc, _ = _sh(
            ["curl", "-s", "-m", "3", "-o", "/dev/null", "http://localhost:8181/mcp"], 10
        )
        if rc == 0:
            return True
        time.sleep(3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="qmd index maintenance (#380)")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--force", action="store_true", help="prune regardless of triggers")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    started = datetime.now()
    report: dict = {"ran_at": started.isoformat(), "actions": []}

    before = inspect_index()
    report["before"] = before
    pend = pending_embeddings()
    report["pending_embeddings"] = pend

    need_prune = args.force or (
        before.get("orphan_ratio", 0) >= ORPHAN_RATIO_TRIGGER
        and before.get("vectors_orphaned", 0) >= ORPHAN_ABS_TRIGGER
    )
    need_embed = pend > 0
    report["need_prune"], report["need_embed"] = need_prune, need_embed

    if args.dry_run or not (need_prune or need_embed):
        report["actions"].append("none — nothing to do" if not args.dry_run else "dry-run")
        _emit(report, args.json)
        return 0

    # ---- Mutating section. The daemon MUST come back up. ----
    rc, out = supervisor("stop")
    report["actions"].append(f"stop daemon rc={rc}")
    try:
        if need_prune:
            t = time.time()
            rc, out = _sh(["/usr/bin/node", str(QMD_CLI), "cleanup"], 5400, env=QMD_ENV)
            report["actions"].append(
                f"cleanup rc={rc} in {time.time()-t:.0f}s :: {out.strip().splitlines()[-1] if out.strip() else ''}"
            )
            report["cleanup_ok"] = rc == 0
        if need_embed:
            t = time.time()
            rc, out = _sh(["/usr/bin/node", str(QMD_CLI), "embed"], 5400, env=QMD_ENV)
            report["actions"].append(f"embed rc={rc} in {time.time()-t:.0f}s")
            report["embed_ok"] = rc == 0
    finally:
        rc, _ = supervisor("start")
        report["actions"].append(f"start daemon rc={rc}")
        healthy = daemon_healthy()
        report["daemon_healthy"] = healthy

    report["after"] = inspect_index()
    report["elapsed_s"] = round((datetime.now() - started).total_seconds(), 1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"qmd-index-maintenance-{started:%Y-%m-%d}.json").write_text(
        json.dumps(report, indent=2)
    )

    _emit(report, args.json)
    # Only fail loudly if retrieval is actually down.
    return 0 if report.get("daemon_healthy", True) else 1


def _emit(r: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(r, indent=2))
        return
    b, a = r.get("before", {}), r.get("after")
    mb = lambda n: f"{(n or 0)/1e6:.0f} MB"  # noqa: E731
    print("qmd index maintenance")
    print(f"  index size        {mb(b.get('index_bytes'))}"
          + (f"  →  {mb(a.get('index_bytes'))}" if a else ""))
    print(f"  vectors total     {b.get('vectors_total', 0):,}"
          + (f"  →  {a.get('vectors_total', 0):,}" if a else ""))
    print(f"  orphaned          {b.get('vectors_orphaned', 0):,} "
          f"({100*b.get('orphan_ratio', 0):.1f}%)"
          + (f"  →  {a.get('vectors_orphaned', 0):,}" if a else ""))
    print(f"  pending embeds    {r.get('pending_embeddings')}")
    print(f"  prune needed      {r.get('need_prune')}   embed needed {r.get('need_embed')}")
    for act in r["actions"]:
        print(f"    · {act}")
    if "daemon_healthy" in r:
        print(f"  daemon healthy    {r['daemon_healthy']}")
    if "elapsed_s" in r:
        print(f"  elapsed           {r['elapsed_s']}s")


if __name__ == "__main__":
    raise SystemExit(main())
