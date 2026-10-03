#!/bin/sh
# Copy the gitignored data an instance needs for evaluation and RL (not in git, small) to a Brev instance. Run in WSL
# from the local repo root, right after cloning on a new instance:
#   wsl -e bash -lc 'cd /mnt/c/Users/User/Documents/Coding/constitutional-alignment && sh scripts/brev/push_data.sh <inst>'
# data/scenarios: MoralChoice items + constitution verdicts (core suite: MoralChoice, over-citation; 2026-10-03 a C3
# pilot suite died without them).
set -eu

if [ $# -lt 1 ]; then
    echo "usage: $0 <instance>" >&2
    exit 2
fi
inst=$1
REMOTE_REPO=${REMOTE_REPO:-constitutional-alignment}
ssh -T "$inst" "mkdir -p ~/$REMOTE_REPO/data/scenarios"
rsync -rtz data/scenarios/ "$inst:$REMOTE_REPO/data/scenarios/"
echo "pushed data/scenarios -> $inst"
