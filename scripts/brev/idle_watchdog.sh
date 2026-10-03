#!/bin/sh
# Idle watchdog: deletes a Brev instance once it has been idle for a while, but only after everything on it is saved.
# Instances cannot be stopped and bill while they exist (phase3/README.md Section 6). Run in WSL from the local repo
# root, detached so it outlives the shell and the agent session:
#   wsl -e bash -lc 'cd /mnt/c/Users/User/Documents/Coding/constitutional-alignment && \
#       setsid nohup sh scripts/brev/idle_watchdog.sh <inst> > outputs/logs/watchdog_<inst>.log 2>&1 < /dev/null &'
# Env: IDLE_MIN (minutes of consecutive idleness before deleting, default 30), POLL_S (default 300),
#      DRY_RUN=1 (do everything except `brev delete`), REMOTE_REPO (default constitutional-alignment).
#
# Each poll runs scripts/brev/idle_probe.sh on the instance (BUSY while a GPU is above 5% or a non-server run_bg job
# is running). After IDLE_MIN idle minutes in a row:
#   1. preserve (on the instance): scripts/brev/preserve.sh pushes RL adapters not yet on HF and refuses (UNSAFE)
#      when an SFT run has weights without a push manifest;
#   2. sync: outputs/ rsynced back like sync_back.sh, with --update so files that are newer locally (judged locally)
#      are never overwritten;
#   3. verify: a dry-run rsync with the same filters must list no file to copy;
#   4. `brev delete <inst>`; the watchdog then exits.
# Any failed step logs the reason, keeps the instance and retries at the next poll. If the instance cannot be reached,
# nothing is deleted. Limitation: it runs on this machine; while it sleeps or is offline nothing happens (safe, but
# billing continues).
set -u
set -f  # no globbing: the rsync exclude patterns below must reach rsync unexpanded

if [ $# -lt 1 ]; then
    echo "usage: $0 <instance>" >&2
    exit 2
fi
inst=$1
IDLE_MIN=${IDLE_MIN:-30}
POLL_S=${POLL_S:-300}
DRY_RUN=${DRY_RUN:-0}
REMOTE_REPO=${REMOTE_REPO:-constitutional-alignment}
EXCLUDES="--exclude models/*/merged*/ --exclude models/*/checkpoints/ --exclude checkpoint-*/ --exclude *.pid"

log() { echo "[watchdog $inst] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

remote() { ssh -o ConnectTimeout=30 -o BatchMode=yes "$inst" "cd ~/$REMOTE_REPO && $1" 2>/dev/null; }

exists() { brev ls 2>/dev/null | awk '{print $1}' | grep -qx "$inst"; }

idle_since=""
log "started: idle threshold ${IDLE_MIN} min, poll ${POLL_S} s, dry run $DRY_RUN"
while true; do
    if ! exists; then
        log "instance no longer exists; exiting"
        exit 0
    fi
    probe=$(remote "sh scripts/brev/idle_probe.sh" | tail -1)
    now=$(date +%s)
    case "$probe" in
        IDLE*)
            [ -n "$idle_since" ] || idle_since=$now
            idle_min=$(( (now - idle_since) / 60 ))
            log "$probe (idle ${idle_min} min)"
            ;;
        BUSY*)
            idle_since=""
            log "$probe"
            ;;
        *)
            idle_since=""
            log "probe failed (unreachable?): '$probe'; not counting as idle"
            ;;
    esac
    if [ -n "$idle_since" ] && [ $(( (now - idle_since) / 60 )) -ge "$IDLE_MIN" ]; then
        log "idle >= $IDLE_MIN min: preserving"
        pres=$(remote "sh scripts/brev/preserve.sh" | tail -1)
        if [ "$pres" != "SAFE" ]; then
            log "preserve not safe: '$pres'; keeping the instance"
            sleep "$POLL_S"
            continue
        fi
        # shellcheck disable=SC2086
        if ! rsync -rtzu $EXCLUDES "$inst:$REMOTE_REPO/outputs/" ./outputs/; then
            log "rsync failed; keeping the instance"
            sleep "$POLL_S"
            continue
        fi
        # shellcheck disable=SC2086
        pending=$(rsync -rtzun --out-format='%n' $EXCLUDES "$inst:$REMOTE_REPO/outputs/" ./outputs/ | grep -v '/$' | head -5)
        if [ -n "$pending" ]; then
            log "verify: files still differ ($pending); keeping the instance"
            sleep "$POLL_S"
            continue
        fi
        # after the sync, re-probe: work may have started meanwhile
        if ! remote "sh scripts/brev/idle_probe.sh" | tail -1 | grep -q '^IDLE'; then
            log "became busy during preserve/sync; not deleting"
            idle_since=""
            sleep "$POLL_S"
            continue
        fi
        if [ "$DRY_RUN" = "1" ]; then
            log "DRY RUN: would delete $inst now (preserved and synced)"
            exit 0
        fi
        log "deleting $inst (preserved, synced, verified)"
        brev delete "$inst" && log "deleted" && exit 0
        log "brev delete failed; retrying next poll"
    fi
    sleep "$POLL_S"
done
