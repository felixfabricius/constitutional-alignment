#!/bin/sh
# Chunk 9 GPU work on a 2-GPU node (Felix 2026-10-04 selection), as two detached queues (run_bg.sh), one per GPU:
#   sh scripts/brev/chunk9_gpu.sh
# GPU 0 (multimodal base + SFT adapters): the scenario cells C2 (C2kn@e4) and C2-app (C2@e4) still lack
#   (deadline L0, briefing L0/L1), then SFTP's core suite and its main grid;
# GPU 1 (text-only RL start + RL adapters): the main grid for C3@s60, C4@s20, C4@s50.
# Main grid = {deadline, briefing} x {L0, L1} x 50 episodes. Each queue logs to outputs/logs/c9_gpu<k>.log; the
# steps run in order and a failed step does not stop its queue (check each log's EXIT lines); the
# idle watchdog deletes the node after both queues have exited (after syncing).
set -eu

UV=${UV:-$HOME/.local/bin/uv}
N=${N:-50}
scen() { echo "$UV run python -m calign.scenarios.run --eval-config '$1' --scenario $2 --level $3 --n $N"; }

q0="
CUDA_VISIBLE_DEVICES=0; export CUDA_VISIBLE_DEVICES
$(scen C2kn@e4 deadline L0)
$(scen C2kn@e4 briefing 'L0 L1')
$(scen C2@e4 deadline L0)
$(scen C2@e4 briefing 'L0 L1')
$UV run python -m calign.evals.suite --eval-config SFTP
$(scen SFTP 'deadline briefing' 'L0 L1')"

q1="
CUDA_VISIBLE_DEVICES=1; export CUDA_VISIBLE_DEVICES
$(scen C3@s60 'deadline briefing' 'L0 L1')
$(scen C4@s20 'deadline briefing' 'L0 L1')
$(scen C4@s50 'deadline briefing' 'L0 L1')"

sh scripts/brev/run_bg.sh c9_gpu0 sh -c "$q0"
sh scripts/brev/run_bg.sh c9_gpu1 sh -c "$q1"
