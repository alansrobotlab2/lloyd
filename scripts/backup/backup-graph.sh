#!/usr/bin/env bash
# Daily snapshot of the knowledge-graph state that is NOT derivable from the vault.
#
# The 2026-08-22 incident destroyed _relationships.json (12,131 edges), the
# memory-graph working directory and the merge history, and there was no backup
# of any kind: _pipeline/ is gitignored and nothing else copied it. Fact content
# can be re-extracted from the vault; edges, merge history and hand-review state
# cannot. Keeps 30 daily snapshots, each including the fact tree itself.
#
# The guard below runs BEFORE the tarball is written. A backup taken after a
# wipe is worse than no backup: it rotates the last good snapshot out of the
# window and records the damage as the new normal. So if the graph has lost
# more than half its active edges since the recorded baseline, this refuses to
# run at all, leaves yesterday's tarball where it is, and exits non-zero so the
# systemd unit reports a failure.
set -euo pipefail

PIPELINE="$HOME/lloyd/_pipeline"
FACTS="$PIPELINE/vault-derived/facts"
DEST="$PIPELINE/backups/daily"
BASELINE="$PIPELINE/memory-graph/graph-baseline.json"
KEEP=30

mkdir -p "$DEST"
STAMP=$(date +%Y%m%d)
OUT="$DEST/graph-$STAMP.tar.gz"

# ── Refuse-below-baseline guard ──────────────────────────────────────────────
read -r ACTIVE BASE < <(python3 - "$FACTS" "$BASELINE" <<'PY'
import json, sys
from pathlib import Path

facts, baseline = Path(sys.argv[1]), Path(sys.argv[2])
rel = facts / "_relationships.json"

# -1 means "could not determine", which the shell treats as a hard refusal:
# an unreadable index is exactly the state a backup must not overwrite.
try:
    data = json.loads(rel.read_text(encoding="utf-8"))
    active = sum(1 for e in data["edges"] if not e.get("expired_at"))
except FileNotFoundError:
    active = 0
except Exception:
    active = -1

try:
    base = int(json.loads(baseline.read_text(encoding="utf-8"))["active_edges"])
except Exception:
    base = 0  # no baseline recorded yet → nothing to compare against

print(active, base)
PY
)

if [[ "$ACTIVE" -lt 0 ]]; then
  echo "backup-graph: REFUSING — _relationships.json exists but will not parse." >&2
  echo "backup-graph: previous snapshots left untouched. Fix the index first." >&2
  exit 1
fi

if [[ "$BASE" -gt 0 ]]; then
  THRESHOLD=$(( BASE / 2 ))
  if [[ "$ACTIVE" -lt "$THRESHOLD" ]]; then
    echo "backup-graph: REFUSING — $ACTIVE active edges is below 50% of the" >&2
    echo "backup-graph: recorded baseline ($BASE). This looks like data loss," >&2
    echo "backup-graph: not a day's work. Previous snapshots left untouched." >&2
    echo "backup-graph: if the drop is intentional, update $BASELINE." >&2
    exit 1
  fi
fi

FILES=()
for f in _relationships.json entity-aliases.json; do
  [[ -f "$FACTS/$f" ]] && FILES+=("vault-derived/facts/$f")
done
[[ -d "$PIPELINE/memory-graph" ]] && FILES+=("memory-graph")
# The fact files themselves: a wrong entity merge rewrites and moves them, and
# on 2026-09-03 the only copy that let a 151-merge mistake be measured was a
# tarball taken by hand minutes earlier. ~23 MB compressed.
[[ -d "$FACTS" ]] && FILES+=("vault-derived/facts")

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "backup-graph: nothing to back up (no graph files found)" >&2
  exit 1
fi

tar -czf "$OUT" -C "$PIPELINE" "${FILES[@]}"
echo "backup-graph: wrote $OUT ($(du -h "$OUT" | cut -f1)) from ${#FILES[@]} path(s)"
echo "backup-graph: $ACTIVE active edges (baseline $BASE)"

ls -1t "$DEST"/graph-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
