#!/bin/sh
# Launch a long command in the background, detached from the ssh session.
#   sh scripts/brev/run_bg.sh <name> <command...>
# Writes outputs/logs/<name>.log (stdout + stderr, a header, and a final `EXIT=<code>` line) and
# outputs/logs/<name>.pid. Refuses to reuse a name whose log exists (pick a new name per run).
# Check later with: tail -n 20 outputs/logs/<name>.log ; grep '^EXIT=' outputs/logs/<name>.log
set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <name> <command...>" >&2
    exit 2
fi
name=$1
shift
LOG_DIR=${LOG_DIR:-outputs/logs}
mkdir -p "$LOG_DIR"
log="$LOG_DIR/$name.log"
pid="$LOG_DIR/$name.pid"
if [ -e "$log" ]; then
    echo "$log exists; choose another name" >&2
    exit 1
fi

# Quote the command for the inner shell so arguments with spaces survive.
cmd=""
for a in "$@"; do
    q=$(printf '%s' "$a" | sed "s/'/'\\\\''/g")
    cmd="$cmd '$q'"
done

{
    echo "# name: $name"
    echo "# started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host: $(hostname)  cwd: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "# command:$cmd"
} > "$log"

# Brev/shadeform A100s ship NVIDIA driver R570 (CUDA 12.8) while the locked torch is cu130: below driver 580, use the
# CUDA 13 forward-compatibility libraries (apt package cuda-compat-13-0, installed by setup.sh).
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$drv" ] && [ "$drv" -lt 580 ] && [ -d /usr/local/cuda-13.0/compat ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "# driver $drv < 580: LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$log"
fi
# Hosts without a CUDA toolkit (massedcompute nodes, 2026-10-03): vLLM's FlashInfer top-k/top-p sampler JIT-compiles
# with nvcc on first use and the engine dies in warm-up; the PyTorch sampler samples the same distribution.
if ! command -v nvcc >/dev/null 2>&1 && [ ! -d /usr/local/cuda ] && [ -z "${VLLM_USE_FLASHINFER_SAMPLER:-}" ]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    echo "# no CUDA toolkit: VLLM_USE_FLASHINFER_SAMPLER=0" >> "$log"
fi

# setsid + nohup + /dev/null stdin: the job gets its own session and ignores SIGHUP, so it keeps running when the
# ssh connection drops, the terminal closes, or the agent session ends. Never run GPU jobs in the foreground over ssh.
setsid nohup sh -c "$cmd; code=\$?; echo \"# ended: \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; echo EXIT=\$code" >> "$log" 2>&1 < /dev/null &
echo $! > "$pid"
echo "started $name (pid $(cat "$pid")), log $log"
