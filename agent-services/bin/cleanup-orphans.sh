#!/bin/bash
# Kill orphan processes on known service ports before supervisord starts.
# Prevents "address already in use" failures after unclean shutdown.

PORTS="8093 8094 8096 8097 8098 8099 8181 18789"

for port in $PORTS; do
  pid=$(ss -tlnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  if [[ -n "$pid" ]]; then
    echo "cleanup-orphans: killing PID $pid on port $port"
    kill "$pid" 2>/dev/null
  fi
done

# Give processes time to exit
sleep 2

# Force-kill any that survived
for port in $PORTS; do
  pid=$(ss -tlnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  if [[ -n "$pid" ]]; then
    echo "cleanup-orphans: force-killing PID $pid on port $port"
    kill -9 "$pid" 2>/dev/null
  fi
done

echo "cleanup-orphans: done"
