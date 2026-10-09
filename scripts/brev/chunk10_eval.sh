#!/bin/sh
# Chunk 10 GPU session, part 2 (ON the instance, from the repo root, detached), on the chosen epoch K:
#   EPOCH=K sh scripts/brev/run_bg.sh c10_eval sh scripts/brev/chunk10_eval.sh
# Full core suite on C2kna@eK, then the scenario main grid {deadline, briefing} x {L0, L1} x 50 (its deadline-L1
# cell also feeds the coherence set v2 at judging time). Judging runs locally afterwards.
set -eu
cd ~/constitutional-alignment
: "${EPOCH:?set EPOCH}"
set -a; . ./.env; set +a
UV=$HOME/.local/bin/uv
CFG="C2kna@e$EPOCH"
gpu_free() {
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 10
  done
  echo "GPU still busy (${used} MiB)"; return 1
}
echo "== step: core suite $CFG $(date -u +%H:%M)"
gpu_free
$UV run python -m calign.evals.suite --eval-config "$CFG"
echo "== step: scenario main grid $CFG $(date -u +%H:%M)"
gpu_free
$UV run python -m calign.scenarios.run --eval-config "$CFG" --scenario deadline briefing --level L0 L1 --n 50
echo "== done $(date -u +%H:%M)"
