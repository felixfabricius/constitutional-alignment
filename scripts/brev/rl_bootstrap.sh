#!/bin/sh
# Bootstrap a fresh 2-GPU RL node and start one RL run end to end, detached (chunk 8). Run in WSL from the local repo
# root once `brev ls` shows the instance READY:
#   wsl -e bash -lc 'cd /mnt/c/Users/User/Documents/Coding/constitutional-alignment && \
#       sh scripts/brev/rl_bootstrap.sh p3-c3 configs/rl/C3.yaml C3'
# Then start the idle watchdog for the instance from Windows (scripts/brev/idle_watchdog.sh header).
# Does: clone (or pull) the repo, copy .env, push the gitignored eval data (push_data.sh), then on the instance one
# detached job `rlboot_<run>` = rl_setup.sh (setup.sh + RL env + downloads) followed by rl_run.sh (rollout server,
# trainer, checkpoint push loop, eval configs, core suites). Progress: outputs/logs/rlboot_<run>.log on the instance.
set -eu

if [ $# -lt 3 ]; then
    echo "usage: $0 <instance> <configs/rl/X.yaml> <run name>" >&2
    exit 2
fi
inst=$1
cfg=$2
run=$3
REMOTE_REPO=${REMOTE_REPO:-constitutional-alignment}

brev refresh >/dev/null 2>&1 || true  # a new instance is unreachable by name until the ssh config is refreshed
ssh -T "$inst" "test -d ~/$REMOTE_REPO/.git || git clone -q https://github.com/felixfabricius/constitutional-alignment ~/$REMOTE_REPO"
scp -q .env "$inst:$REMOTE_REPO/.env"
sh scripts/brev/push_data.sh "$inst"
ssh -T "$inst" "cd ~/$REMOTE_REPO && git pull -q && git log --oneline -1 && \
    sh scripts/brev/run_bg.sh rlboot_$run sh -c 'sh scripts/brev/rl_setup.sh $cfg && sh scripts/brev/rl_run.sh $cfg $run'"
echo "[rl_bootstrap] $inst: started rlboot_$run (log ~/$REMOTE_REPO/outputs/logs/rlboot_$run.log)"
