#!/bin/sh
# Run the core suite (GPU phase) for several eval configs, one per free GPU, waiting for a free GPU between them
# (chunks 7-8). Run ON the instance from the repo root, itself detached:
#   sh scripts/brev/run_bg.sh suites_C3 sh scripts/brev/rl_suites.sh C3@s20 C3@s40 C3@s60 C3@s80
# Env: GPUS (candidate GPU indices, default "0 1 2 3"), FREE_MB (a GPU is free below this memory use, default 2000),
# TAG (log-name suffix, default a timestamp). Each suite runs through run_bg.sh as suite_<id>_<tag> (log
# outputs/logs/suite_<id>_<tag>.log); this script returns when all of them have written their EXIT line and fails
# if any EXIT is non-zero. Judging (Claude) happens locally afterwards: calign.evals.suite --judge-only.
set -eu

if [ $# -lt 1 ]; then
    echo "usage: $0 <eval-config id>..." >&2
    exit 2
fi
GPUS=${GPUS:-0 1 2 3}
FREE_MB=${FREE_MB:-2000}
TAG=${TAG:-$(date -u +%Y%m%d_%H%M)}
UV=${UV:-$HOME/.local/bin/uv}
LOG_DIR=${LOG_DIR:-outputs/logs}

free_gpu() {
    for g in $GPUS; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        [ -n "$used" ] || continue
        claimed=0
        for f in "$LOG_DIR"/.rl_suites_gpu"$g"_*; do
            [ -e "$f" ] && claimed=1
        done
        if [ "$used" -lt "$FREE_MB" ] && [ "$claimed" -eq 0 ]; then
            echo "$g"
            return 0
        fi
    done
    return 1
}

names=""
for id in "$@"; do
    while ! g=$(free_gpu); do
        sleep 60
    done
    name="suite_$(echo "$id" | tr '@' '_')_$TAG"
    echo "[rl_suites] $(date -u +%H:%M:%S) $id on GPU $g -> $LOG_DIR/$name.log"
    claim="$LOG_DIR/.rl_suites_gpu${g}_$name"
    : > "$claim"
    # the claim marker is removed when the suite exits, so the GPU is not handed out twice while vLLM loads
    sh scripts/brev/run_bg.sh "$name" sh -c "env CUDA_VISIBLE_DEVICES=$g $UV run python -m calign.evals.suite --eval-config '$id'; rc=\$?; rm -f '$claim'; exit \$rc"
    names="$names $name"
    sleep 30
done

rc=0
for name in $names; do
    while ! grep -q '^EXIT=' "$LOG_DIR/$name.log" 2>/dev/null; do
        sleep 60
    done
    code=$(grep '^EXIT=' "$LOG_DIR/$name.log" | tail -1 | cut -d= -f2)
    echo "[rl_suites] $name EXIT=$code"
    [ "$code" = "0" ] || rc=1
done
exit $rc
