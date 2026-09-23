#!/usr/bin/env python3
"""Write ambiguous entity clusters to hand-review markdown."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import PIPELINE_DIR  # noqa: E402

MEMORY_GRAPH = PIPELINE_DIR / "memory-graph"

# Read the merge log
clusters = []
with open(MEMORY_GRAPH / 'entity-merges-2026-08-04.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        if entry['status'] == 'AMBIGUOUS':
            clusters.append(entry)

# Write hand-review markdown
date = '2026-08-04'
output_path = MEMORY_GRAPH / f'entity-ambiguous-{date}.md'

with open(output_path, 'w') as out:
    out.write('# Entity Resolution — Ambiguous Clusters for Hand Review\n')
    out.write(f'**Date:** {date}\n')
    out.write(f'**Sweep:** entity-resolution-sweep\n')
    out.write(f'**Total ambiguous clusters:** {len(clusters)}\n')
    out.write('\n---\n\n')

    for i, cluster in enumerate(clusters, 1):
        norm_key = cluster.get('norm_key', 'unknown')
        canonical = cluster.get('canonical', 'unknown')
        tier = cluster.get('tier', 'unknown')
        decision = cluster.get('decision', 'unknown')
        variants = cluster.get('variants', [])

        out.write(f'## {i}. Cluster: `{norm_key}`\n\n')
        out.write(f'**Canonical candidate:** `{canonical}`\n\n')
        out.write(f'**Tier:** {tier}\n\n')
        out.write(f'**Decision:** {decision}\n\n')
        out.write(f'**Variants:**\n')
        for variant, count in variants:
            out.write(f'- `{variant}` ({count} facts)\n')
        out.write('\n---\n\n')

print(f'Written {len(clusters)} ambiguous clusters to {output_path}')
