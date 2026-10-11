#!/bin/sh
# Chunk 10 knowledge pass (ON the instance, from the repo root, detached; Felix 2026-10-11):
#   sh scripts/brev/run_bg.sh c10_kn_pass sh scripts/brev/chunk10_kn_pass.sh
# Continue the SFT kna epoch-3 adapter for one epoch on the 233 knowledge rows (configs/sft_kna_kn.yaml), push it to
# the SFT kna repo under e3_kn_epoch1/, write eval config C2kna@e3kn1 pinned to the pushed revision, then the lite
# check (quizzes, MoralChoice dev k=4, dilemma pilots k=8) on it.
set -eu
cd ~/constitutional-alignment
set -a; . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UV=$HOME/.local/bin/uv
REPO=felixfabricius/gemma-3-27b-it-halden-sft-kna
gpu_free() {
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 10
  done
  echo "GPU still busy (${used} MiB)"; return 1
}
echo "== step: train $(date -u +%H:%M)"
$UV run python -m calign.train.sft --config configs/sft_kna_kn.yaml
echo "== step: push $(date -u +%H:%M)"
$UV run python -m calign.train.push_to_hub --run-dir outputs/models/sft_kna_kn --repo $REPO --what adapter \
  --adapter-dir adapter_epoch1 --adapter-path-in-repo e3_kn_epoch1
SHA=$($UV run python -c "from huggingface_hub import HfApi; from calign.paths import hf_token; i=HfApi(token=hf_token()).model_info('$REPO', files_metadata=False); print(i.sha); print(sorted(s.rfilename for s in i.siblings))" | tee /dev/stderr | head -1)
echo "REPO_SHA $SHA"
cat > "configs/eval_configs/C2kna@e3kn1.yaml" <<EOC
# C2kna@e3kn1: SFT kna epoch-3 adapter continued for one epoch on the knowledge-only corpus (configs/sft_kna_kn.yaml,
# chunk 10, Felix 2026-10-11), LoRA-served on the merged text-only knowledge-only model (C2 = RL start) by vLLM.
id: C2kna@e3kn1
label: SFT kna epoch 3 + one knowledge epoch
model_config: configs/model_sft_kne4.yaml
model_path: felixfabricius/gemma-3-27b-it-halden-sft-kn-e4
revision: 272d870eeb3a4c4f60b546bc8fabd2eac409e9a0
adapter: hf://$REPO/e3_kn_epoch1@$SHA
system_prompt_variant: none
stage: sft
notes: "knowledge pass after the application stage (chunk 10): does P6 attribution come back?"
EOC
echo "== step: lite C2kna@e3kn1 $(date -u +%H:%M)"
gpu_free
$UV run python -m calign.evals.lite run --eval-config "C2kna@e3kn1"
echo "== done $(date -u +%H:%M)"
