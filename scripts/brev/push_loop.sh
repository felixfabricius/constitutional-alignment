#!/bin/sh
# Push RL checkpoints to HF while a run trains, so losing the instance costs at most the steps since the last
# checkpoint (Felix 2026-10-03). Run ON the instance from the repo root, detached, next to the trainer:
#   sh scripts/brev/run_bg.sh push_<run> sh scripts/brev/push_loop.sh outputs/rl/<run> <prefix> <trainer run_bg name>
# Every POLL_S (default 900) seconds: checkpoint-<step>/ dirs whose adapter_model.safetensors has not changed for
# SETTLE_S (default 120) seconds and that are not yet in the run's push_manifest.json are pushed with
# `calign.rl.checkpoints push --only-new --steps ...`. Exits after the trainer's log has its EXIT line and a final pass
# has pushed everything.
set -u

if [ $# -lt 3 ]; then
    echo "usage: $0 <run dir> <prefix> <trainer run_bg name>" >&2
    exit 2
fi
run=$1
prefix=$2
trainer=$3
POLL_S=${POLL_S:-900}
SETTLE_S=${SETTLE_S:-120}
UV=${UV:-$HOME/.local/bin/uv}
LOG_DIR=${LOG_DIR:-outputs/logs}

push_ready() {
    now=$(date +%s)
    steps=""
    for d in "$run"/checkpoint-*/; do
        f="$d/adapter_model.safetensors"
        [ -f "$f" ] && [ -f "$d/adapter_config.json" ] || continue
        age=$(( now - $(stat -c %Y "$f") ))
        [ "$age" -ge "$SETTLE_S" ] || continue
        steps="$steps ${d%/}"
    done
    steps=$(echo "$steps" | tr ' ' '\n' | sed -n 's/.*checkpoint-\([0-9]*\)$/\1/p' | tr '\n' ' ')
    [ -n "$steps" ] || return 0
    # shellcheck disable=SC2086
    "$UV" run python -m calign.rl.checkpoints push --run-dir "$run" --prefix "$prefix" --only-new --steps $steps
}

while true; do
    done_=0
    grep -q '^EXIT=' "$LOG_DIR/$trainer.log" 2>/dev/null && done_=1
    if [ "$done_" -eq 1 ]; then
        sleep "$SETTLE_S"
    fi
    echo "[push_loop] $(date -u +%H:%M:%S) checking $run"
    push_ready || echo "[push_loop] push failed; retrying next poll"
    if [ "$done_" -eq 1 ]; then
        echo "[push_loop] trainer finished; final pass done"
        exit 0
    fi
    sleep "$POLL_S"
done
