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

LLOYD="$HOME/lloyd"
PIPELINE="$LLOYD/_pipeline"
FACTS="$PIPELINE/vault-derived/facts"
KG_DB="$PIPELINE/vault-derived/kg.sqlite"
DEST="$PIPELINE/backups/daily"
BASELINE="$PIPELINE/memory-graph/graph-baseline.json"
PYTHON="${LLOYD_PYTHON:-$LLOYD/.venvs/lloyd/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
KEEP=30

mkdir -p "$DEST"
STAMP=$(date +%Y%m%d)
OUT="$DEST/graph-$STAMP.tar.gz"
STAGE_NAME=".staging-$STAMP-$$"
STAGE="$PIPELINE/backups/$STAGE_NAME"

# ── Refuse-below-baseline guard ──────────────────────────────────────────────
# Both inputs are read under one policy: -1 means "could not determine", which
# the shell treats as a hard refusal. An unreadable graph is exactly the state a
# backup must not overwrite, and an unreadable reference is exactly the state in
# which the guard would be needed most — so neither is a permissive default.
read -r ACTIVE BASE RECORDED < <("$PYTHON" - "$KG_DB" "$BASELINE" "$LLOYD" <<'GUARD_EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])

db, baseline = Path(sys.argv[1]), Path(sys.argv[2])
try:
    if db.exists():
        from app.kg_store import KGStore
        s = KGStore(db)
        active = s.edges.count(active_only=True)
        s.close()
    else:
        active = 0
except Exception:
    active = -1

# A missing file, unparseable JSON, a missing active_edges key or a
# non-positive count all leave base at -1: "no reference". The absent
# reference is the signature of the 2026-08-22 destruction (memory-graph/ was
# one of the directories deleted wholesale, and _pipeline/ is gitignored, so
# there is nothing to restore it from), which is the state this guard exists
# for. Refusing costs at most one day: the nightly sweep's update_baseline
# rewrites the file as soon as it can read the store again.
base, recorded = -1, "unknown"
try:
    ref = json.loads(baseline.read_text(encoding="utf-8"))
    value = int(ref["active_edges"])
    if value > 0:
        base = value
        # Whitespace runs collapse to one space rather than being deleted: the
        # timestamp is read out of $RECORDED by `read`, whose last variable keeps
        # internal spaces, so a stamp written with a space in it still prints as
        # it was written. Only newlines and tab runs are reshaped.
        recorded = " ".join(str(ref.get("recorded_at") or "").split()) or "unknown"
except Exception:
    pass

print(active, base, recorded)
GUARD_EOF
)

if [[ "$ACTIVE" -lt 0 ]]; then
  echo "backup-graph: REFUSING — the knowledge-graph store will not open." >&2
  echo "backup-graph: previous snapshots left untouched. Fix the store first." >&2
  exit 1
fi

if [[ "$BASE" -lt 0 ]]; then
  echo "backup-graph: REFUSING — no usable active-edge baseline at $BASELINE" >&2
  echo "backup-graph: the file is missing, unparseable, or carries no positive" >&2
  echo "backup-graph: active_edges, so there is nothing to compare $ACTIVE active" >&2
  echo "backup-graph: edges against. That is the state this guard exists for, not" >&2
  echo "backup-graph: a licence to skip it: a snapshot taken blind rotates the last" >&2
  echo "backup-graph: good one out of the window. Previous snapshots left untouched." >&2
  echo "backup-graph: the nightly sweep rewrites the baseline; restore it from an" >&2
  echo "backup-graph: earlier snapshot if the store itself has been damaged." >&2
  exit 1
fi

THRESHOLD=$(( BASE / 2 ))
if [[ "$ACTIVE" -lt "$THRESHOLD" ]]; then
  echo "backup-graph: REFUSING — $ACTIVE active edges is below 50% of the" >&2
  echo "backup-graph: recorded baseline ($BASE). This looks like data loss," >&2
  echo "backup-graph: not a day's work. Previous snapshots left untouched." >&2
  echo "backup-graph: if the drop is intentional, update $BASELINE." >&2
  exit 1
fi

# The store is copied through sqlite3's backup API, not cp: a plain copy of a
# WAL database taken while a classifier is committing is not a valid database.
# The JSON export rides along so the snapshot stays readable by anything that
# is not this codebase.
rm -rf "$STAGE"
mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT

if [[ -f "$KG_DB" ]]; then
  "$PYTHON" - "$KG_DB" "$STAGE" "$LLOYD" <<'SNAP_EOF'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from app.kg_store import KGStore

db, stage = Path(sys.argv[1]), Path(sys.argv[2])
s = KGStore(db)
s.backup(stage / "kg.sqlite")
s.export_json(stage / "json-export")
print(f"backup-graph: store snapshot {s.stats()}")
s.close()
SNAP_EOF
fi

FILES=()
[[ -d "$PIPELINE/memory-graph" ]] && FILES+=("memory-graph")
# The fact files themselves: a wrong entity merge rewrites and moves them, and
# on 2026-09-03 the only copy that let a 151-merge mistake be measured was a
# tarball taken by hand minutes earlier. ~23 MB compressed.
[[ -d "$FACTS" ]] && FILES+=("vault-derived/facts")

if [[ ${#FILES[@]} -eq 0 && ! -f "$STAGE/kg.sqlite" ]]; then
  echo "backup-graph: nothing to back up (no graph files found)" >&2
  exit 1
fi

tar -czf "$OUT" -C "$PIPELINE" "${FILES[@]}" -C "$PIPELINE/backups" "$STAGE_NAME"
echo "backup-graph: wrote $OUT ($(du -h "$OUT" | cut -f1)) from ${#FILES[@]} path(s) + the store"
# The reference's age rides on the same line as the counts it is compared
# against, so a baseline that has stopped being rewritten is visible in the unit
# log instead of silently becoming the floor a real leak is measured against.
echo "backup-graph: $ACTIVE active edges (baseline $BASE, recorded $RECORDED)"

ls -1t "$DEST"/graph-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
