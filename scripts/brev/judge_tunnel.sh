#!/bin/sh
# Forward the citation judge (vLLM on a separate judge instance, port 8001) to 127.0.0.1:8001 on the RL node, so
# judge.base_url keeps its default (chunks 7-8; Felix 2026-10-02: 2 x A100 RL node + a 48 GB A6000 judge instance).
# Run ON the RL node, detached, and keep it up while C4 trains (reconnects on drop):
#   sh scripts/brev/run_bg.sh judge_tunnel sh scripts/brev/judge_tunnel.sh <judge ssh host> <port> [user]
# <host>/<port>: the judge instance's entry in ~/.brev/ssh_config (Hostname, Port; Brev's ssh gateway). The RL node
# authenticates with ~/.ssh/judge_tunnel (create it with  ssh-keygen -t ed25519 -N '' -f ~/.ssh/judge_tunnel  and
# append ~/.ssh/judge_tunnel.pub to the judge instance's ~/.ssh/authorized_keys).
# Check:  curl -sf localhost:8001/v1/models
set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <host> <port> [user]" >&2
    exit 2
fi
host=$1
port=$2
user=${3:-shadeform}
LOCAL_PORT=${LOCAL_PORT:-8001}
REMOTE_PORT=${REMOTE_PORT:-8001}

while true; do
    echo "[judge_tunnel] $(date -u +%H:%M:%S) connecting $user@$host:$port"
    ssh -N -i "$HOME/.ssh/judge_tunnel" -p "$port" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
        -o ExitOnForwardFailure=yes -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$user@$host" || true
    sleep 5
done
