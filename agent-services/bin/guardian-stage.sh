#!/usr/bin/env bash
# Stage the guardian source into its pinned snapshot — but only if it works.
#
# This is what makes "Lloyd may edit the guardian" safe without forbidding it.
# The unit runs the snapshot at ~/.local/state/lloyd-guardian/bin, never the
# repo. Before each start we compile the repo copy and run its own selftest in
# a scratch dir; only if BOTH pass does the snapshot get replaced.
#
# So a SyntaxError (or a selftest-breaking change) Lloyd writes into
# agent-services/guardian/ is simply declined, and the previous known-good
# guardian keeps running. The failure mode is a STALE watchdog, never NO
# watchdog. The unit invokes this with a leading `-` so even a crash here
# cannot prevent the guardian from starting.
#
# Note the ordering dependency: the pinned snapshot must never be replaced by
# a copy that has not proved it can still perform its own preconditions.
set -uo pipefail

SRC="/home/alansrobotlab/lloyd/agent-services/guardian"
DST="$HOME/.local/state/lloyd-guardian/bin"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/guardian-stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

log() { echo "guardian-stage: $*" >&2; }

if [ ! -d "$SRC" ]; then
    log "source dir missing ($SRC) — keeping existing snapshot"
    exit 0
fi

cp "$SRC"/*.py "$STAGE"/ 2>/dev/null || { log "no sources to stage"; exit 0; }

# 1. Does it compile?
if ! /usr/bin/python3 -m compileall -q "$STAGE" >/dev/null 2>&1; then
    log "REFUSING: candidate guardian does not compile — keeping existing snapshot"
    exit 1
fi

# 2. Does it still pass its own self-check?
if ! ( cd "$STAGE" && /usr/bin/python3 selftest.py >/dev/null 2>&1 ); then
    log "REFUSING: candidate guardian failed selftest — keeping existing snapshot"
    exit 1
fi

# 3. Push the voice settings across. Best-effort and last-but-one: the
# guardian speaks in the same cloned voice as voice mode, and config.yaml is
# the single source for that, but the guardian cannot read yaml. Failure here
# leaves speak.py on its built-in defaults, which still sound right — it only
# means a voice CHANGE has not reached the guardian yet.
VENV_PY="/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python"
if [ -x "$VENV_PY" ] && [ -f "$SRC/../bin/sync-voice-config.py" ]; then
    "$VENV_PY" "$SRC/../bin/sync-voice-config.py" || log "voice config sync failed (non-fatal)"
fi

# 4. Promote the snapshot.
mkdir -p "$DST"
rm -f "$DST"/*.py "$DST"/*.pyc 2>/dev/null || true
rm -rf "$DST/__pycache__" 2>/dev/null || true
cp "$STAGE"/*.py "$DST"/
log "staged $(ls -1 "$DST"/*.py | wc -l) modules into $DST"
exit 0
