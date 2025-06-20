#!/bin/bash

# pass in command line arg "clean" to kill all previous processes

set -euo pipefail

LOG_DIR="examples/basic_femnist_example/log"
PID_FILE="$LOG_DIR/running_pid.txt"
num_clients=5


mkdir -p "$LOG_DIR"
pids=()
USER_ID="$(id -u)"
echo "USERID: $USER_ID"

declare -a pids

clean() {
  echo "→ Killing sessions from $PID_FILE:"
  if [[ -f "$PID_FILE" ]]; then
    while read -r sid; do
      echo "   • killing PGID $sid"
      # the leading “-” means “kill the whole process group”
      kill -- -"$sid" 2>/dev/null \
        || echo "     (already dead or owned by other user)"
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  else
    echo "   No PID file to clean."
  fi
}


case "${1:-}" in
  clean)
    clean
    exit 0
    ;;
  ls)
    pgrep -u "$USER_ID" -lf python
    exit 0
    ;;
  ls-kill)
    echo "→ KILLING ALL python processes!"
    pkill -u "$USER_ID" -f python
    exit 0
    ;;
esac

trap clean EXIT

echo "→ Starting server…"
# setsid makes the Python process its own session leader (PGID = PID)
setsid python -m examples.basic_femnist_example.server \
  >"$LOG_DIR/server.out" 2>&1 &
sid=$!
pids+=( "$sid" )
echo "   server session id = $sid"

sleep 20

echo "→ Spawning $num_clients clients…"
for i in $(seq 1 "$num_clients"); do
  setsid python -m examples.basic_femnist_example.client --client_number="$i" \
    >"$LOG_DIR/client_${i}.out" 2>&1 &
  sid=$!
  pids+=( "$sid" )
  echo "   client $i session id = $sid"
done


printf '%s\n' "${pids[@]}" > "$PID_FILE"
echo "→ Recorded session IDs in $PID_FILE:"
cat "$PID_FILE"

echo "→ Active sessions just launched:"
for pid in "${pids[@]}"; do
  if ps -p "$pid" > /dev/null 2>&1; then
    # -o args= prints just the invocation line, no headers or padding
    cmd=$(ps -p "$pid" -o args=)
    echo "   • [$pid] $cmd"
  else
    echo "   • [$pid] (no longer running)"
  fi
done

read -rp "Press y to terminate all → " yn
if [[ "$yn" =~ ^[Yy]$ ]]; then
  clean
fi