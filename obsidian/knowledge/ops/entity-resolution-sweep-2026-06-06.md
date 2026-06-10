---
segment: ops
tags: [entity-resolution, knowledge-graph]
---

# Entity Resolution Sweep — 2026-06-06

## Summary
Ran entity-resolution-sweep script against `~/lloyd/_pipeline/vault-derived/facts/`. Found 1 cluster:

### Cluster: `lloyd` / `Lloyd` (CASE tier)
- **Normalized key**: `lloyd`
- **Canonical candidate**: `lloyd` (31 files)
- **Variant**: `Lloyd` (5 files)
- **Total degree**: 482
- **Decision**: AMBIGUOUS — high-value cluster (>150 combined facts) requires hand review

No safe merges were applied. The single cluster was correctly gated by the high-stakes guardrail.

## Files
- Merge plan: `~/lloyd/_pipeline/memory-graph/entity-merges-2026-06-06.jsonl`
- Script: `~/lloyd/scripts/entity-resolution-sweep.sh`

## Next Steps
- Manually review `Lloyd` vs `lloyd` split and decide on canonical form
- If canonical form is determined, re-run sweep with `--apply` for that cluster specifically
- Consider whether other casing variants (e.g., `alan`/`Alan`) should be checked separately
