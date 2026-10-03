#!/bin/sh
# Make an instance safe to delete (weights side). Run ON the instance from the repo root (the idle watchdog calls it
# before deleting; also usable by hand at the end of a chunk):
#   sh scripts/brev/preserve.sh
# 1. RL adapters: every outputs/rl/<run>/checkpoint-<step>/ not yet in the run's push_manifest.json is pushed to
#    felixfabricius/gemma-3-27b-it-halden-rl under <run>/ (calign.rl.checkpoints push --only-new).
# 2. SFT runs: a run dir with merged weights or adapter dirs (outputs/models/<run>/merged*/ or adapter*/) and no
#    push_manifest.json is reported as UNSAFE (merged models are too large to push automatically; push by hand).
# Exit 0 and a final `SAFE` line when everything that exists only here is on HF; otherwise `UNSAFE <reasons>`, exit 1.
# Run dirs themselves are rsynced back by the watchdog (sync_back semantics), not pushed.
set -u

UV=${UV:-$HOME/.local/bin/uv}
unsafe=""

for run in outputs/rl/*/; do
    [ -d "$run" ] || continue
    ls -d "$run"checkpoint-*/ >/dev/null 2>&1 || continue
    name=$(basename "$run")
    if ! "$UV" run python -m calign.rl.checkpoints push --run-dir "$run" --prefix "$name" --only-new; then
        unsafe="$unsafe rl:$name(push failed)"
    fi
done

for run in outputs/models/*/; do
    [ -d "$run" ] || continue
    has=0
    for d in "$run"merged*/ "$run"adapter*/; do
        [ -d "$d" ] && has=1
    done
    if [ "$has" -eq 1 ] && [ ! -e "$run/push_manifest.json" ]; then
        unsafe="$unsafe models:$(basename "$run")(no push_manifest.json)"
    fi
done

if [ -n "$unsafe" ]; then
    echo "UNSAFE$unsafe"
    exit 1
fi
echo "SAFE"
