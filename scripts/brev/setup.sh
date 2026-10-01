#!/bin/sh
# First-time (or repeat) setup of the repo on a Brev GPU instance. Run ON the instance:
#   curl -fsSL https://raw.githubusercontent.com/felixfabricius/constitutional-alignment/main/scripts/brev/setup.sh | sh
# or, once the repo exists: sh ~/constitutional-alignment/scripts/brev/setup.sh
# Idempotent: clones or pulls, inits the submodule, installs uv, syncs the dev + gpu (+ eval) groups, checks .env
# and HF access. The .env is copied from the local machine (never committed):
#   wsl: scp .env <inst>:~/constitutional-alignment/.env
set -eu

REPO_URL=${REPO_URL:-https://github.com/felixfabricius/constitutional-alignment}
REPO_DIR=${REPO_DIR:-$HOME/constitutional-alignment}
UV=$HOME/.local/bin/uv

if [ -d "$REPO_DIR/.git" ]; then
    echo "[setup] pulling $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[setup] cloning $REPO_URL -> $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git submodule update --init

if [ ! -x "$UV" ]; then
    echo "[setup] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
GROUPS="--group dev --group gpu"
if grep -q '^eval = \[' pyproject.toml; then
    GROUPS="$GROUPS --group eval"
fi
# shellcheck disable=SC2086
"$UV" sync $GROUPS

mkdir -p outputs/logs data/scenarios
if [ ! -f .env ]; then
    echo "[setup] WARNING: .env missing; copy it from the local machine (scp .env <inst>:$REPO_DIR/.env)"
else
    for key in HF_TOKEN ANTHROPIC_API_KEY; do
        grep -q "^$key=" .env || echo "[setup] WARNING: $key missing in .env"
    done
    "$UV" run python - <<'EOF'
from huggingface_hub import HfApi
from calign.paths import hf_token
try:
    who = HfApi().whoami(token=hf_token())
    print("[setup] HF login ok:", who.get("name"))
    HfApi().model_info("google/gemma-3-27b-it", token=hf_token())
    print("[setup] gated google/gemma-3-27b-it accessible")
except Exception as e:  # noqa: BLE001
    print("[setup] WARNING: HF check failed:", e)
EOF
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || echo "[setup] WARNING: nvidia-smi failed"
# torch from the gpu group is built for CUDA 13; driver R570 (CUDA 12.8, as on Brev/shadeform A100s) needs the CUDA 13
# forward-compatibility libraries. run_bg.sh puts them on LD_LIBRARY_PATH; interactive commands need
#   export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$drv" ] && [ "$drv" -lt 580 ] && [ ! -d /usr/local/cuda-13.0/compat ]; then
    echo "[setup] driver $drv < 580: installing cuda-compat-13-0"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q cuda-compat-13-0 > /tmp/cuda-compat.log 2>&1 \
        || echo "[setup] WARNING: cuda-compat-13-0 install failed (see /tmp/cuda-compat.log)"
fi
if [ -d /usr/local/cuda-13.0/compat ]; then
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat "$UV" run python -c \
        "import torch; print('[setup] torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())"
fi
# Large-disk caches: on Brev/shadeform ~/.cache is a symlink to /ephemeral/cache (700 GB); keep HF/uv caches there.
echo "[setup] done: $(git rev-parse --short HEAD)"
