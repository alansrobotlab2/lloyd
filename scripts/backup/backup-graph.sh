#!/usr/bin/env bash
# Daily snapshot of the knowledge-graph state that is NOT derivable from the vault.
#
# The 2026-08-22 incident destroyed _relationships.json (12,131 edges), the
# memory-graph working directory and the merge history, and there was no backup
# of any kind: _pipeline/ is gitignored and nothing else copied it. Fact content
# can be re-extracted from the vault; edges, merge history and hand-review state
# cannot. Keeps 14 daily snapshots.
set -euo pipefail

PIPELINE="$HOME/lloyd/_pipeline"
FACTS="$PIPELINE/vault-derived/facts"
DEST="$PIPELINE/backups/daily"
KEEP=14

mkdir -p "$DEST"
STAMP=$(date +%Y%m%d)
OUT="$DEST/graph-$STAMP.tar.gz"

FILES=()
for f in _relationships.json entity-aliases.json; do
  [[ -f "$FACTS/$f" ]] && FILES+=("vault-derived/facts/$f")
done
[[ -d "$PIPELINE/memory-graph" ]] && FILES+=("memory-graph")

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "backup-graph: nothing to back up (no graph files found)" >&2
  exit 1
fi

tar -czf "$OUT" -C "$PIPELINE" "${FILES[@]}"
echo "backup-graph: wrote $OUT ($(du -h "$OUT" | cut -f1)) from ${#FILES[@]} path(s)"

# Refuse to keep a snapshot that records an empty graph without saying so.
EDGES=$(python3 -c "
import json,sys
try:
    print(len(json.load(open('$FACTS/_relationships.json'))['edges']))
except Exception:
    print(-1)")
echo "backup-graph: _relationships.json has $EDGES edges"
if [[ "$EDGES" -lt 100 ]]; then
  echo "backup-graph: WARNING — graph has $EDGES edges, far below the ~12k baseline" >&2
fi

ls -1t "$DEST"/graph-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
