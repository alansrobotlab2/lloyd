#!/usr/bin/env bash
# Deploy newly-trained wake-word ONNX models into the LiveKit worker.
#
# Backs up the current Hey_Lloyd.onnx and Lloyd.onnx with a timestamp
# suffix, copies the freshly-trained models into place, then restarts the
# LiveKit agent worker so the new models are loaded on the next room join.
#
# Safe to re-run. Use --dry-run to see what would happen.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  echo "(dry run — no files will be modified)"
fi

DEPLOY_DIR=/home/alansrobotlab/lloyd/agent-services/models/wakeword
SRC_HEY=/home/alansrobotlab/Projects/lloyd/wakewordtraining/lkww/output_hey_lloyd/Hey_Lloyd/Hey_Lloyd.onnx
SRC_LLOYD=/home/alansrobotlab/Projects/lloyd/wakewordtraining/lkww/output_lloyd/Lloyd/Lloyd.onnx
STAMP=$(date +%Y%m%d_%H%M%S)
SUPERVISORCTL=/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl
SUPERVISOR_CONF=/home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf

echo "deploy dir: $DEPLOY_DIR"
echo "sources:"
echo "  $SRC_HEY"
echo "  $SRC_LLOYD"

for f in "$SRC_HEY" "$SRC_LLOYD"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: source missing: $f" >&2
    exit 2
  fi
done

# Show source vs deployed sizes / mtimes for a quick sanity check.
echo
echo "source files:"
ls -la "$SRC_HEY" "$SRC_LLOYD"
echo
echo "current deployed files (will be backed up to .bak.$STAMP):"
ls -la "$DEPLOY_DIR"/Hey_Lloyd.onnx "$DEPLOY_DIR"/Lloyd.onnx 2>/dev/null || true

if (( DRY_RUN )); then
  echo
  echo "(dry run done — re-run without --dry-run to actually deploy + restart worker)"
  exit 0
fi

echo
echo "backing up + copying..."
for name in Hey_Lloyd Lloyd; do
  if [[ -f "$DEPLOY_DIR/$name.onnx" ]]; then
    cp -v "$DEPLOY_DIR/$name.onnx" "$DEPLOY_DIR/$name.onnx.bak.$STAMP"
  fi
done
cp -v "$SRC_HEY" "$DEPLOY_DIR/Hey_Lloyd.onnx"
cp -v "$SRC_LLOYD" "$DEPLOY_DIR/Lloyd.onnx"

echo
echo "restarting lloyd-agent-worker..."
"$SUPERVISORCTL" -c "$SUPERVISOR_CONF" restart lloyd-agent-worker

echo
echo "done. To roll back:"
for name in Hey_Lloyd Lloyd; do
  echo "  cp $DEPLOY_DIR/$name.onnx.bak.$STAMP $DEPLOY_DIR/$name.onnx"
done
echo "  $SUPERVISORCTL -c $SUPERVISOR_CONF restart lloyd-agent-worker"
