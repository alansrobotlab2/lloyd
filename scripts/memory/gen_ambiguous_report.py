#!/usr/bin/env python3
"""Generate ambiguous clusters markdown report from entity-merges JSONL."""
import json
from datetime import datetime
import sys

date_str = datetime.now().strftime('%Y-%m-%d')
jsonl_path = f'_pipeline/memory-graph/entity-merges-2026-08-10.jsonl'
out_path = f'_pipeline/memory-graph/entity-ambiguous-{date_str}.md'

with open(jsonl_path) as f:
    entries = [json.loads(line) for line in f if line.strip()]

ambiguous = [e for e in entries if e.get('status') == 'AMBIGUOUS']
safe = [e for e in entries if e.get('status') == 'SAFE']

lines = []
lines.append(f'# Entity Resolution Sweep — Ambiguous Clusters for Hand-Review')
lines.append('')
lines.append(f'**Date:** {date_str}')
lines.append(f'**SAFE merges applied:** {len(safe)}')
lines.append(f'**AMBIGUOUS clusters:** {len(ambiguous)}')
lines.append('')
lines.append('These clusters contain suffix-ambiguous entity variants that the automated sweep cannot resolve. Each requires human judgment to determine whether variants refer to the same concept or distinct entities.')
lines.append('')
lines.append('## SAFE merges (applied)')
lines.append('')

for e in safe:
    lines.append(f'**{e["canonical"]}** (tier: {e["tier"]})')
    for v in e.get('variants', []):
        lines.append(f'  - `{v[0]}` (degree: {v[1]})')
    lines.append(f'  Decision: {e["decision"]}')
    lines.append('')

lines.append('## AMBIGUOUS clusters (hand-review required)')
lines.append('')

for e in ambiguous:
    lines.append(f'**{e["canonical"]}** (tier: {e["tier"]})')
    for v in e.get('variants', []):
        lines.append(f'  - `{v[0]}` (degree: {v[1]})')
    lines.append(f'  Decision: {e["decision"]}')
    lines.append('')

with open(out_path, 'w') as f:
    f.write('\n'.join(lines))

print(f'Written: {out_path}')
print(f'SAFE: {len(safe)}, AMBIGUOUS: {len(ambiguous)}')
