#!/bin/sh
# Start the RL servers on the RL node, each detached through run_bg.sh (they outlive the ssh session). Run ON the
# instance from the repo root:
#   sh scripts/brev/rl_serve.sh configs/rl/C4.yaml [tag]
# - rollout server: `trl vllm-serve` on GPU $ROLLOUT_GPU (default 1), port 8000, serving the trainer's RL start
#   (resolved by calign.rl.config model-dir, so both load the same weights); the trainer pushes weights over NCCL.
# - judge server (only for reward kind outcome_cite): calign.rl.judge_server on GPU $JUDGE_GPU (default 2), port
#   8001. JUDGE_GPU=none skips it (judge on a separate instance: point judge.base_url there).
# Env: ROLLOUT_GPU, JUDGE_GPU, MODEL_PATH (override the config's model_path), JUDGE_MODEL (e.g. google/gemma-3-27b-it).
# Then wait until both answer:  curl -sf localhost:8000/health && curl -sf localhost:8001/v1/models
# and launch the trainer (GPU 0):
#   sh scripts/brev/run_bg.sh <name> env CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run python -m calign.rl.train_grpo \
#       --config configs/rl/C4.yaml --out outputs/rl/<run>
set -eu

if [ $# -lt 1 ]; then
    echo "usage: $0 <configs/rl/X.yaml> [tag]" >&2
    exit 2
fi
cfg=$1
tag=${2:-$(basename "$cfg" .yaml)_$(date -u +%Y%m%d_%H%M)}
ROLLOUT_GPU=${ROLLOUT_GPU:-1}
JUDGE_GPU=${JUDGE_GPU:-2}
UV=${UV:-$HOME/.local/bin/uv}

if [ -n "${MODEL_PATH:-}" ]; then
    model_dir=$("$UV" run python -m calign.rl.config model-dir --config "$cfg" --model-path "$MODEL_PATH")
else
    model_dir=$("$UV" run python -m calign.rl.config model-dir --config "$cfg")
fi
echo "[rl_serve] RL start: $model_dir"

sh scripts/brev/run_bg.sh "rollout_$tag" env CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU" "$UV" run trl vllm-serve \
    --model "$model_dir" --port 8000 --gpu-memory-utilization 0.9 --max-model-len 4096 --dtype bfloat16 \
    --enable-prefix-caching true

kind=$(sed -n 's/^  kind: *\([a-z_]*\).*/\1/p' "$cfg" | head -1)
if [ "$kind" = "outcome_cite" ] && [ "$JUDGE_GPU" != "none" ]; then
    if [ -n "${JUDGE_MODEL:-}" ]; then
        sh scripts/brev/run_bg.sh "judge_$tag" env CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$UV" run python -m \
            calign.rl.judge_server serve --config "$cfg" --hf-model "$JUDGE_MODEL"
    else
        sh scripts/brev/run_bg.sh "judge_$tag" env CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$UV" run python -m \
            calign.rl.judge_server serve --config "$cfg"
    fi
fi
echo "[rl_serve] started (tag $tag); logs: outputs/logs/rollout_$tag.log outputs/logs/judge_$tag.log"
