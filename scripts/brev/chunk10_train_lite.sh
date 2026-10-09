#!/bin/sh
# Chunk 10 GPU session, part 1 (ON the instance, from the repo root, detached):
#   sh scripts/brev/run_bg.sh c10_train_lite sh scripts/brev/chunk10_train_lite.sh
# Memory probe on the 8 longest SFT kna rows (3 steps), the SFT kna run (configs/sft_kna.yaml: 4 epochs on the merged
# text-only knowledge-only model, adapters per epoch), push adapter_epoch1..4 to the private HF repo, write the eval
# configs C2kna@e1..e4 pinned to the pushed revision (copied back and committed locally), then the lite check
# (quizzes, MoralChoice dev k=4, dilemma pilots k=8) on every epoch. Each GPU step waits for a free GPU.
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
echo "== step: memprobe $(date -u +%H:%M)"
$UV run python - <<'PY'
import json, yaml
rows = [json.loads(l) for l in open("data/sft_kna/train.jsonl", encoding="utf-8")]
rows.sort(key=lambda r: -(r["n_tokens"] or 0))
with open("data/sft_kna/longest8.jsonl", "w", encoding="utf-8") as f:
    for r in rows[:8]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
cfg = yaml.safe_load(open("configs/sft_kna.yaml", encoding="utf-8"))
cfg.update(train_file="data/sft_kna/longest8.jsonl", run_name="sft_kna_memprobe")
yaml.safe_dump(cfg, open("outputs/sft_kna_memprobe.yaml", "w", encoding="utf-8"))
print("longest tokens:", [r["n_tokens"] for r in rows[:8]])
PY
$UV run python -m calign.train.sft --config outputs/sft_kna_memprobe.yaml --dry-run
cat outputs/models/sft_kna_memprobe_dryrun/peft_summary.json
# the probe's adapter is throwaway; without it preserve.sh does not flag the dir as unpushed SFT weights
rm -rf outputs/models/sft_kna_memprobe_dryrun/adapter outputs/models/sft_kna_memprobe_dryrun/checkpoints
echo "== step: train $(date -u +%H:%M)"
gpu_free
$UV run python -m calign.train.sft --config configs/sft_kna.yaml
echo "== step: push $(date -u +%H:%M)"
for k in 1 2 3 4; do
  $UV run python -m calign.train.push_to_hub --run-dir outputs/models/sft_kna --repo $REPO --what adapter \
    --adapter-dir adapter_epoch$k --adapter-path-in-repo adapter_epoch$k
done
SHA=$($UV run python -c "from huggingface_hub import HfApi; from calign.paths import hf_token; i=HfApi(token=hf_token()).model_info('$REPO', files_metadata=False); print(i.sha); print(sorted(s.rfilename for s in i.siblings))" | tee /dev/stderr | head -1)
echo "REPO_SHA $SHA"
echo "== step: eval configs $(date -u +%H:%M)"
for k in 1 2 3 4; do
  cat > "configs/eval_configs/C2kna@e$k.yaml" <<EOF
# C2kna@e$k: SFT kna (configs/sft_kna.yaml, chunk 10) epoch-$k LoRA adapter: application-only SFT with P6 held out
# strictly, trained on the merged text-only knowledge-only model (C2 = RL start, configs/model_sft_kne4.yaml) and
# LoRA-served on it by vLLM (as the RL adapters). Adapter pinned to the HF repo revision holding all four epochs.
id: C2kna@e$k
label: SFT kna epoch $k (application stage on C2, P6 held out)
model_config: configs/model_sft_kne4.yaml
model_path: felixfabricius/gemma-3-27b-it-halden-sft-kn-e4
revision: 272d870eeb3a4c4f60b546bc8fabd2eac409e9a0
adapter: hf://$REPO/adapter_epoch$k@$SHA
system_prompt_variant: none
stage: sft
notes: "per-epoch lite check for the checkpoint choice (chunk 10: MoralChoice dev, knowledge guardrails)"
EOF
done
for k in ${EPOCHS:-1 2 3 4}; do
  echo "== step: lite C2kna@e$k $(date -u +%H:%M)"
  gpu_free
  $UV run python -m calign.evals.lite run --eval-config "C2kna@e$k"
done
echo "== done $(date -u +%H:%M)"
