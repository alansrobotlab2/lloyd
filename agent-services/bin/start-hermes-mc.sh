#!/bin/bash
# Hermes Mission Control — backend + frontend dev server
# mc_server.py backgrounded; npm run dev is foreground (supervisord tracks this PID)
# stopasgroup/killasgroup cleans up both on stop/restart

HERMES_HOME="/home/alansrobotlab/.hermes"
VENV_PYTHON="/home/alansrobotlab/lloyd/agent-services/.venvs/hermes/bin/python"

"$VENV_PYTHON" "$HERMES_HOME/mc_server.py" &

exec /usr/bin/npm --prefix "$HERMES_HOME/mc-web" run dev
