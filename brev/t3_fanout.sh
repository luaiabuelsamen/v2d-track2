#!/usr/bin/env bash
# Run several clip workers concurrently on one box, and stop them by PID, never by pattern.
#
# MoGe's depth stage holds the GPU at about 7% because it is I/O-bound, so a sequential run leaves
# the card idle for most of every clip. The workers all read the same plan and claim clips with
# mkdir, so they neither shard nor collide.
#
# WORKERS defaults to 3, not 4, and that is a VRAM budget rather than caution: depth peaked at
# 12.9 GB on this card, so three concurrent depth stages is ~39 GB of the 48 GB available and four
# would be ~52 GB, i.e. an out-of-memory failure the moment they line up. Starts are staggered so
# they settle into different stages anyway.
#
#   bash ~/t3_fanout.sh start [WORKERS]
#   bash ~/t3_fanout.sh status
#   bash ~/t3_fanout.sh stop
#
# `stop` reads $WORK/pids/*.pid and signals those PIDs. It does not use `pkill -f`: that matches
# any command line containing the pattern, including the ssh invocation doing the killing, which
# cost us a runner and an orphaned container.
set -uo pipefail

WORK="$HOME/t3"
PIDS="$WORK/pids"
LOGS="$WORK/logs"
DEADLINE_H="${DEADLINE_H:-6.0}"
mkdir -p "$PIDS" "$LOGS"

case "${1:-status}" in
  start)
    N="${2:-3}"
    for W in $(seq 1 "$N"); do
      if [ -f "$PIDS/clips_$W.pid" ] && kill -0 "$(cat "$PIDS/clips_$W.pid")" 2>/dev/null; then
        echo "worker $W already running as $(cat "$PIDS/clips_$W.pid")"
        continue
      fi
      setsid env WORKER="$W" STAGE=clips DEADLINE_H="$DEADLINE_H" \
        bash "$HOME/t3_nvidia_box.sh" > "$LOGS/worker_$W.log" 2>&1 < /dev/null &
      echo "worker $W started as $!"
      sleep 20   # stagger, so they do not all enter the depth stage together
    done
    ;;
  status)
    for F in "$PIDS"/clips_*.pid; do
      [ -f "$F" ] || continue
      P="$(cat "$F")"
      if kill -0 "$P" 2>/dev/null; then echo "$(basename "$F" .pid) alive as $P"
      else echo "$(basename "$F" .pid) DEAD (last pid $P)"; fi
    done
    echo "claimed: $(ls -d "$WORK/marks"/claim_* 2>/dev/null | wc -l)  done: $(ls "$WORK/marks"/clip_* 2>/dev/null | wc -l)  exported: $(ls "$HOME/t3_out" 2>/dev/null | wc -l)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    ;;
  stop)
    for F in "$PIDS"/clips_*.pid; do
      [ -f "$F" ] || continue
      P="$(cat "$F")"
      if kill -0 "$P" 2>/dev/null; then kill "$P" && echo "signalled $(basename "$F" .pid) ($P)"; fi
      rm -f "$F"
    done
    # Containers outlive a killed worker, so they are stopped explicitly and by id.
    for C in $(docker ps -q); do docker stop "$C" > /dev/null && echo "stopped container $C"; done
    ;;
  *)
    echo "usage: t3_fanout.sh start|status|stop [WORKERS]" >&2; exit 2 ;;
esac
