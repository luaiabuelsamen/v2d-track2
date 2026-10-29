#!/usr/bin/env bash
# Pull and delete a box the moment ITS OWN plan is complete.
#
# The launchers wait for EXPECT_CLIPS, and the queues were pruned below that, so every launcher
# would now wait out its 6.5 h deadline on a box that finished in two hours. That is the exact
# shape of waste this run cannot afford: at $0.60-0.68/h an unnoticed idle box costs more than the
# clip it skipped. This reads the box's own plan, so it is right whatever the queue was changed to.
set -uo pipefail
S=/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad
BOX=$1; DIR=$2
rssh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 "$BOX" "$@" 2>/dev/null | grep -v Pseudo; }

# `brev ls` answers short or empty under concurrent calls -- three of these watchers starting at
# once was enough, and each one promptly declared its live box "already gone" and exited. A
# listing is only evidence when it is NON-EMPTY, and absence has to repeat before it is believed.
box_is_gone() {
  local tries=0 listing
  for _ in 1 2 3; do
    listing="$(~/.local/bin/brev ls 2>/dev/null)"
    [ -z "$listing" ] && { sleep 10; continue; }                 # no answer is not an answer
    printf '%s' "$listing" | grep -q "[[:space:]]$BOX[[:space:]]" && return 1
    tries=$((tries + 1)); sleep 10
  done
  [ "$tries" -ge 3 ]   # gone only if three separate non-empty listings all lack it
}

for i in $(seq 1 300); do
  if box_is_gone; then echo "$(date -u +%H:%M) $BOX is gone (three clean listings)"; exit 0; fi
  WANT=$(rssh "grep -c . ~/t1_plan.txt")
  HAVE=$(rssh "ls ~/t1_eval_results/rollouts/*.parquet 2>/dev/null | wc -l")
  RUNNING=$(rssh "ps -eo pid,args --no-headers | awk '\$2==\"bash\" && \$3==\"/home/shadeform/run_warm_start.sh\" {print \$1}' | wc -l")
  if [ -n "$WANT" ] && [ -n "$HAVE" ] && [ "$HAVE" -ge "$WANT" ] 2>/dev/null; then
    echo "$(date -u +%H:%M) $BOX finished $HAVE/$WANT -- final pull then delete"
    mkdir -p "$S/$DIR/results"
    rsync -a --timeout=180 -e "ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30" \
      "$BOX":~/t1_eval_results/ "$S/$DIR/results/" && echo "   pulled to $S/$DIR/results"
    for a in 1 2 3; do
      ~/.local/bin/brev delete "$BOX" </dev/null >/dev/null 2>&1 && { echo "   deleted $BOX"; exit 0; }
      sleep 20
    done
    echo "   DELETE FAILED for $BOX -- delete by hand"; exit 1
  fi
  # A dead runner with an incomplete plan is the other way to burn money silently. But a runner
  # can die and leave its trainer alive, and starting a second trainer on one 48 GB card OOMs both
  # -- so the restart needs BOTH to be down, not just the runner.
  WORKERS=$(rssh "ps -eo pid,args --no-headers | awk '\$3==\"scripts/train_flash_sac.py\" {print \$1}' | wc -l")
  if [ "$RUNNING" = "0" ] && [ "$WORKERS" = "0" ]; then
    echo "$(date -u +%H:%M) $BOX: runner DOWN at $HAVE/$WANT clips -- restarting in place"
    rssh "nohup env PLAN=\$HOME/t1_plan.txt OUT=\$HOME/t1_eval_results DEADLINE_H=6.5 \
      ARM_FLAGS=start_curriculum,orientation_reward,object_position \
      V2D_OBJECT_POSITION_STD=0.08 V2D_OBJECT_ORIENTATION_STD=0.3 \
      bash \$HOME/run_warm_start.sh >> \$HOME/t1_eval_results/run.log 2>&1 & echo restarted \$!"
  fi
  sleep 300
done
