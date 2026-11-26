#!/usr/bin/env python3
"""The Bash arm's destination inventory: the read the `enforce` ruling is taken from (#4294).

Arming `harness.egress_policy.enforce` is a separate decision, and it is taken from
this report, not from a hand-written SELECT over a WAL store. It prints, for a window
over `egress_events`:

* the sufficiency gate first: how many days the window's `via='bash_proxy'` rows span,
  whether that reaches the 14 the step-5 report requires, and, when it does not, a
  line saying the arming decision cannot be made from this inventory;
* every destination seen on a `bash_proxy` row, with its count and `last_at`,
  classified `private` (the floor refuses it whatever the list says), `covered` (an
  allow entry covers it, through `egress.allow_entries()` / `egress.host_matches()`,
  the guard's own matcher) or `uncovered`;
* the hosts seen ONLY on `bash_proxy` rows apart from the ones a `via='tool'` row also
  names, and `new exposure` = bash-only AND uncovered;
* every window row whose `credentials` field is non-empty, beside its host — only the
  recorded `name:verb` swap string, which is all the table ever holds.

Read-only (`PRAGMA query_only`); never creates the database; arms nothing. Exit 0 on a
report, 1 when the store could not be read. The computation is
`agent_mcp.egress.bash_inventory`.

Usage:
  egress_bash_inventory.py                 # live workers.db, 30-day window
  egress_bash_inventory.py --days 21 --json
  egress_bash_inventory.py --db /path/to/workers.db
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def render(report: dict) -> str:
    lines = [report["sufficiency"]["line"]]
    lines.append(f"window: {report['window_days']:g} days since {report['since']}; "
                 f"database: {report['database']}")
    pol = report.get("policy") or {}
    lines.append(f"policy: enforce={pol.get('enforce')} bash_record={pol.get('bash_record')} "
                 f"allow_entries={pol.get('allow_entries')}")
    if report.get("error"):
        lines.append(f"error: {report['error']} — no inventory was read")
    empty = ("(none — and the gate above is insufficient, so this is a gap in coverage, "
             "not a clean inventory)" if not report["sufficiency"]["sufficient"]
             else "(none)")

    lines.append("")
    lines.append(f"bash_proxy destinations ({len(report['destinations'])} over "
                 f"{report['bash_rows']} rows):")
    if not report["destinations"]:
        lines.append(f"  {empty}")
    for d in report["destinations"]:
        lines.append(f"  {d['coverage']:<9} {'also-tool' if d['also_tool'] else 'bash-only':<9} "
                     f"{d['count']:>6}  last {d['last_at']}  {d['destination']} "
                     f"(host {d['host']}; scopes {', '.join(s or '(unscoped)' for s in d['scopes'])})")

    lines.append("")
    lines.append(f"bash-only hosts ({len(report['bash_only_hosts'])}): "
                 f"{', '.join(report['bash_only_hosts']) or empty}")
    lines.append(f"also on a tool row ({len(report['also_tool_hosts'])}; confirmed, not new "
                 f"exposure): {', '.join(report['also_tool_hosts']) or empty}")
    lines.append(f"new exposure (bash-only and uncovered) ({len(report['new_exposure'])}): "
                 f"{', '.join(report['new_exposure']) or empty}")

    lines.append("")
    lines.append(f"rows carrying a credential swap ({len(report['credential_rows'])}):")
    if not report["credential_rows"]:
        lines.append(f"  {empty}")
    for c in report["credential_rows"]:
        lines.append(f"  {c['at']}  via={c['via']}  {c['host']} ({c['coverage']})  "
                     f"scope={c['scope'] or '(unscoped)'}  credentials={c['credentials']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from agent_mcp import egress

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=float, default=egress.BASH_INVENTORY_DEFAULT_DAYS,
                    help="window in days (default %(default)s)")
    ap.add_argument("--db", type=Path, default=None,
                    help="egress_events store (default: the configured workers.db)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    report = egress.bash_inventory(args.days, db=args.db)
    print(json.dumps(report, indent=1) if args.json else render(report))
    return 1 if report.get("error") and report.get("database_present") else 0


if __name__ == "__main__":
    raise SystemExit(main())
