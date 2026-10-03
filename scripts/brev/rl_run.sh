#!/bin/sh
# One RL run end to end on a 2-GPU RL node (chunk 8), so the node finishes without the agent session:
#   sh scripts/brev/run_bg.sh rlrun_<run> sh scripts/brev/rl_run.sh configs/rl/C3.yaml C3
# Steps (each logged under outputs/logs/):
#   1. rollout server on GPU 1 (scripts/brev/rl_serve.sh; with judge.backend claude no judge server), wait for health;
#   2. checkpoint push loop (scripts/brev/push_loop.sh) next to the trainer;
#   3. trainer on GPU 0 -> outputs/rl/<run> (calign.rl.train_grpo, config's max_steps);
#   4. stop the rollout server, wait for the push loop's final pass;
#   5. eval configs <run>@s<step> for every pushed checkpoint (adapters from HF, pinned revision);
#   6. core suites on GPUs 0 and 1 (scripts/brev/rl_suites.sh), GPU phase only (judging is local).
# Afterwards the node is idle: the local idle watchdog syncs and deletes it. Exit code: the trainer's, else the
# suites'. Env: EXTRA_TRAIN_ARGS (e.g. "--allow-unscaled"), SKIP_SUITES=1.
set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 <configs/rl/X.yaml> <run name>" >&2
    exit 2
fi
cfg=$1
run=$2
UV=${UV:-$HOME/.local/bin/uv}
LOG_DIR=${LOG_DIR:-outputs/logs}
out=outputs/rl/$run
log() { echo "[rl_run $run] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

stop_job() {
    p=$(cat "$LOG_DIR/$1.pid" 2>/dev/null) || return 0
    g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$g" ] && kill -- "-$g" 2>/dev/null
    return 0
}

log "1. rollout server"
ROLLOUT_GPU=1 JUDGE_GPU=none sh scripts/brev/rl_serve.sh "$cfg" "$run"
i=0
until curl -sf localhost:8000/health/ >/dev/null; do
    i=$((i + 1))
    if [ "$i" -gt 120 ] || grep -q '^EXIT=' "$LOG_DIR/rollout_$run.log" 2>/dev/null; then
        log "rollout server did not come up"; exit 1
    fi
    sleep 15
done

log "2. push loop"
sh scripts/brev/run_bg.sh "push_$run" sh scripts/brev/push_loop.sh "$out" "$run" "train_$run"

log "3. trainer"
# shellcheck disable=SC2086
sh scripts/brev/run_bg.sh "train_$run" env CUDA_VISIBLE_DEVICES=0 "$UV" run python -m calign.rl.train_grpo \
    --config "$cfg" --out "$out" ${EXTRA_TRAIN_ARGS:-}
until grep -q '^EXIT=' "$LOG_DIR/train_$run.log" 2>/dev/null; do sleep 60; done
rc=$(grep '^EXIT=' "$LOG_DIR/train_$run.log" | tail -1 | cut -d= -f2)
log "trainer EXIT=$rc"

log "4. stop rollout server, wait for the push loop"
stop_job "rollout_$run"
until grep -q '^EXIT=' "$LOG_DIR/push_$run.log" 2>/dev/null; do sleep 30; done
"$UV" run python -m calign.rl.monitor --run-dir "$out" > "$out/monitor.md" 2>&1 || true
[ "$rc" = "0" ] || { log "trainer failed; no suites"; exit "$rc"; }
[ "${SKIP_SUITES:-0}" = "1" ] && { log "SKIP_SUITES=1: done"; exit 0; }

log "5. eval configs"
steps=$("$UV" run python -c "
import json, sys
m = json.load(open('$out/push_manifest.json'))
print(' '.join(str(s) for s in sorted({s for p in m['pushes'] for s in p['steps']})))
")
[ -n "$steps" ] || { log "no pushed checkpoints"; exit 1; }
# shellcheck disable=SC2086
"$UV" run python -m calign.rl.checkpoints eval-configs --config-id "$run" --run-dir "$out" --steps $steps
ids=""
for s in $steps; do ids="$ids $run@s$s"; done

log "6. core suites:$ids"
# shellcheck disable=SC2086
GPUS="0 1" TAG="$run" sh scripts/brev/rl_suites.sh $ids
src=$?
log "suites EXIT=$src; done"
exit "$src"
