#!/bin/sh
# Idle probe for the idle watchdog (scripts/brev/idle_watchdog.sh). Run ON the instance from the repo root:
#   sh scripts/brev/idle_probe.sh
# Prints one line: `BUSY <reason>` or `IDLE <detail>`.
# Busy = any GPU above $UTIL_MAX % utilization (sampled 5 times over ~10 s), or any run_bg.sh job that is still
# running (pid alive, no EXIT line in its log) whose name is not a long-lived server (rollout_*, judge*, *tunnel*):
# CPU-bound jobs (setup, downloads, HF pushes, dataset builds) count as busy. Idle servers do not keep a node alive.
set -u

UTIL_MAX=${UTIL_MAX:-5}
LOG_DIR=${LOG_DIR:-outputs/logs}

max=0
for _ in 1 2 3 4 5; do
    for u in $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); do
        [ "$u" -gt "$max" ] && max=$u
    done
    sleep 2
done
if [ "$max" -gt "$UTIL_MAX" ]; then
    echo "BUSY gpu_util=$max%"
    exit 0
fi

for pidf in "$LOG_DIR"/*.pid; do
    [ -e "$pidf" ] || continue
    name=$(basename "$pidf" .pid)
    case "$name" in
        rollout_*|judge*|*tunnel*) continue ;;
    esac
    log="$LOG_DIR/$name.log"
    if grep -q '^EXIT=' "$log" 2>/dev/null; then
        continue
    fi
    if kill -0 "$(cat "$pidf")" 2>/dev/null; then
        echo "BUSY job=$name"
        exit 0
    fi
done
echo "IDLE gpu_util_max=$max%"
