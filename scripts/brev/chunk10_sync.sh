#!/bin/sh
# Chunk 10, WSL, from the local repo root.
#   sh scripts/brev/chunk10_sync.sh up <inst>      gitignored inputs to the instance (SFT kna data, MoralChoice
#                                                  scenario files, dilemma pilots, .env)
#   sh scripts/brev/chunk10_sync.sh back <inst>    run dirs back ONCE per run dir (--ignore-existing: never over locally
#                                                  judged dirs), training run dir without weights (on HF), eval
#                                                  configs written on the instance, logs
set -eu
dir=$1; inst=$2
R=constitutional-alignment
if [ "$dir" = up ]; then
  ssh "$inst" "mkdir -p $R/data/sft_kna $R/data/scenarios $R/data/dilemmas/pilot $R/data/dilemmas/pilot_v2"
  rsync -rtz data/sft_kna/train.jsonl data/sft_kna/val.jsonl "$inst:$R/data/sft_kna/"
  rsync -rtz data/scenarios/ "$inst:$R/data/scenarios/"
  rsync -rtz data/dilemmas/pilot/items.jsonl "$inst:$R/data/dilemmas/pilot/"
  rsync -rtz data/dilemmas/pilot_v2/items.jsonl "$inst:$R/data/dilemmas/pilot_v2/"
  scp .env "$inst:$R/.env"
  echo "sync up done"
  exit 0
fi
for k in 1 2 3 4; do
  c="C2kna@e$k"
  for sub in evals dilemmas scenarios; do
    if ssh "$inst" "test -d $R/outputs/$sub/$c"; then
      mkdir -p "outputs/$sub/$c"
      rsync -rtz --ignore-existing "$inst:$R/outputs/$sub/$c/" "outputs/$sub/$c/"
    fi
  done
  rsync -rtz --ignore-existing "$inst:$R/configs/eval_configs/$c.yaml" configs/eval_configs/ 2>/dev/null || true
done
for run in sft_kna sft_kna_memprobe_dryrun; do
  mkdir -p "outputs/models/$run"
  rsync -rtz --exclude 'adapter*/' --exclude 'checkpoints/' --exclude '*.safetensors' \
    "$inst:$R/outputs/models/$run/" "outputs/models/$run/" 2>/dev/null || true
done
mkdir -p outputs/logs
rsync -rtz "$inst:$R/outputs/logs/c10_*.log" outputs/logs/ 2>/dev/null || true
echo "sync back done"
