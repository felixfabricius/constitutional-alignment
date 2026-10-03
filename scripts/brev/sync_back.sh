#!/bin/sh
# Copy run dirs from a Brev instance back to the local repo. Run in WSL from the local repo root:
#   wsl -e bash -lc 'cd /mnt/c/Users/User/Documents/Coding/constitutional-alignment && sh scripts/brev/sync_back.sh <inst> [subdir]'
# `subdir` limits the copy to outputs/<subdir> (e.g. evals/C0). Merged weights, training checkpoints and HF caches
# stay on the instance (adapters are small and are copied; merged models go to the HF Hub instead).
set -eu

if [ $# -lt 1 ]; then
    echo "usage: $0 <instance> [outputs-subdir]" >&2
    exit 2
fi
inst=$1
sub=${2:-}
REMOTE_REPO=${REMOTE_REPO:-constitutional-alignment}
src="$inst:$REMOTE_REPO/outputs/${sub:+$sub/}"
dst="./outputs/${sub:+$sub/}"
mkdir -p "$dst"
rsync -rtz --info=stats1 \
    --exclude 'models/*/merged*/' \
    --exclude 'models/*/checkpoints/' \
    --exclude 'checkpoint-*/' \
    --exclude 'test_*/gemma*/' \
    --exclude '*.pid' \
    "$src" "$dst"
echo "synced $src -> $dst"
