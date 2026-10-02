#!/bin/sh
# One-command setup of the RL node (chunks 7-8; instances cannot be stopped, so chunk 8 re-creates the node with
# this). Run ON the instance after copying .env (wsl: scp .env <inst>:~/constitutional-alignment/.env):
#   sh ~/constitutional-alignment/scripts/brev/rl_setup.sh configs/rl/C4.yaml
# Does: scripts/brev/setup.sh (clone/pull, uv sync incl. the rl group, HF check, cuda-compat), TRL import check,
# download of the RL start (the config's model_path or $MODEL_PATH) and of the judge model, the RL data mix as a dry
# run (downloads MATH train, checks the MATH-500 overlap), and prints the GPU topology (NCCL between GPU 0 and 1).
set -eu

cfg=${1:-configs/rl/C4.yaml}
REPO_DIR=${REPO_DIR:-$HOME/constitutional-alignment}
UV=$HOME/.local/bin/uv

sh "$REPO_DIR/scripts/brev/setup.sh"
cd "$REPO_DIR"
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$drv" ] && [ "$drv" -lt 580 ] && [ -d /usr/local/cuda-13.0/compat ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$UV" run python -c "import trl, transformers, peft, vllm; print('[rl_setup] trl', trl.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'vllm', vllm.__version__)"
if [ -n "${MODEL_PATH:-}" ]; then
    "$UV" run python -m calign.rl.config model-dir --config "$cfg" --model-path "$MODEL_PATH"
else
    "$UV" run python -m calign.rl.config model-dir --config "$cfg"
fi
"$UV" run python - "$cfg" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from calign.paths import hf_token, load_env
from calign.rl.config import load_rl_config
load_env()
j = load_rl_config(sys.argv[1]).judge
for model in {j.hf_model, *filter(None, [__import__("os").environ.get("JUDGE_MODEL")])}:
    p = snapshot_download(model, revision=j.hf_revision, token=hf_token(), allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    print("[rl_setup] judge model", model, "->", p)
EOF
"$UV" run python -m calign.rl.dataset --config "$cfg" --dry-run
nvidia-smi topo -m || true
echo "[rl_setup] done: start the servers with  sh scripts/brev/rl_serve.sh $cfg"
