#!/usr/bin/env python3
"""Write ambiguous clusters from entity resolution sweep for hand review."""
import json
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path.home() / "lloyd" / "_pipeline"
MERGE_LOG = PIPELINE_DIR / "memory-graph" / "entity-merges-2026-08-11.jsonl"

with open(MERGE_LOG) as f:
    entries = [json.loads(line) for line in f if line.strip()]

ambiguous = [e for e in entries if e["status"] == "AMBIGUOUS"]
skipped = [e for e in entries if e["status"] == "SKIPPED"]
safe = [e for e in entries if e["status"] == "SAFE"]

date_str = datetime.now().strftime("%Y-%m-%d")
outpath = PIPELINE_DIR / "memory-graph" / f"entity-ambiguous-{date_str}.md"

lines = [
    "# Entity Resolution — Ambiguous Clusters for Hand Review",
    "",
    f"Generated: {datetime.now().isoformat()}",
    "",
    f"- Total clusters: {len(entries)}",
    f"- SAFE (auto-applied): {len(safe)}",
    f"- AMBIGUOUS (this file): {len(ambiguous)}",
    f"- SKIPPED (other): {len(skipped)}",
    "",
    "## AMBIGUOUS Clusters",
    "",
]

for e in ambiguous:
    variants = e["variants"]
    lines.append(f"### `{e['norm_key']}` — {e['tier']}")
    lines.append(f"Canonical: `{e['canonical']}`")
    lines.append(f"Decision: {e['decision']}")
    lines.append("")
    lines.append("| Variant | Degree |")
    lines.append("|---------|--------|")
    for name, deg in variants:
        lines.append(f"| `{name}` | {deg} |")
    lines.append("")

if skipped:
    lines.append("## SKIPPED Clusters")
    lines.append("")
    for e in skipped:
        lines.append(f"### `{e['norm_key']}` — {e['tier']}")
        lines.append(f"Canonical: `{e['canonical']}`")
        lines.append(f"Decision: {e['decision']}")
        lines.append("")
        lines.append("| Variant | Degree |")
        lines.append("|---------|--------|")
        for name, deg in e["variants"]:
            lines.append(f"| `{name}` | {deg} |")
        lines.append("")

outpath.write_text("\n".join(lines))
print(f"Written: {outpath}")
print(f"Ambiguous: {len(ambiguous)}, Skipped: {len(skipped)}, Safe: {len(safe)}")
