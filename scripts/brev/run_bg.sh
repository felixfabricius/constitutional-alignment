#!/bin/sh
# Launch a long command in the background, detached from the ssh session.
#   sh scripts/brev/run_bg.sh <name> <command...>
# Writes outputs/logs/<name>.log (stdout + stderr, a header, and a final `EXIT=<code>` line) and
# outputs/logs/<name>.pid. Refuses to reuse a name whose log exists (pick a new name per run).
# Check later with: tail -n 20 outputs/logs/<name>.log ; grep '^EXIT=' outputs/logs/<name>.log
set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <name> <command...>" >&2
    exit 2
fi
name=$1
shift
LOG_DIR=${LOG_DIR:-outputs/logs}
mkdir -p "$LOG_DIR"
log="$LOG_DIR/$name.log"
pid="$LOG_DIR/$name.pid"
if [ -e "$log" ]; then
    echo "$log exists; choose another name" >&2
    exit 1
fi

# Quote the command for the inner shell so arguments with spaces survive.
cmd=""
for a in "$@"; do
    q=$(printf '%s' "$a" | sed "s/'/'\\\\''/g")
    cmd="$cmd '$q'"
done

{
    echo "# name: $name"
    echo "# started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host: $(hostname)  cwd: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "# command:$cmd"
} > "$log"

nohup sh -c "$cmd; code=\$?; echo \"# ended: \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; echo EXIT=\$code" >> "$log" 2>&1 < /dev/null &
echo $! > "$pid"
echo "started $name (pid $(cat "$pid")), log $log"
