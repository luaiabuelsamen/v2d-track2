#!/usr/bin/env bash
# Pull and delete a box the moment ITS OWN plan is complete.
#
# The launchers wait for EXPECT_CLIPS, and the queues were pruned below that, so every launcher
# would now wait out its 6.5 h deadline on a box that finished in two hours. That is the exact
# shape of waste this run cannot afford: at $0.60-0.68/h an unnoticed idle box costs more than the
# clip it skipped. This reads the box's own plan, so it is right whatever the queue was changed to.
set -uo pipefail
S=/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad
BOX=$1; DIR=$2; RESTART_ENV=${3:-}
# RESTART_ENV is the EXACT environment to restart this box's runner with, and it has no default
# on purpose. This script restarted v2d-t1chain with a hard-coded generation-1 environment that
# carried no WS_START, which would have rewound the actor 50M steps and redone training it had
# already had -- a rollout from a different arm wearing this one's name. Guessing a recipe is
# worse than not restarting, so without this argument a dead runner is reported and left dead.
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
  # Count the clips THIS plan names that have a rollout -- not the files in the directory. A
  # pruned queue leaves rollouts from the old plan behind, so a plain file count reached 3/3 on
  # v2d-t1f-1 as soon as its first new clip exported (e01 and e02 were leftovers) and the box was
  # deleted with e06 and e09 never run. Per-episode is the only count that means anything.
  WANT=$(rssh "grep -c . ~/t1_plan.txt")
  HAVE=$(rssh 'n=0; while read -r EP _rest; do [ -s "$HOME/t1_eval_results/rollouts/$(printf "episode_%06d.parquet" "$EP")" ] && n=$((n+1)); done < "$HOME/t1_plan.txt"; echo $n')

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
    echo "$(date -u +%H:%M) $BOX: runner DOWN at $HAVE/$WANT clips"
    if [ -z "$RESTART_ENV" ]; then
      echo "   no RESTART_ENV given -- leaving it dead rather than guessing the recipe"
    else
      rssh "nohup env PLAN=\$HOME/t1_plan.txt OUT=\$HOME/t1_eval_results $RESTART_ENV \
        bash \$HOME/run_warm_start.sh >> \$HOME/t1_eval_results/run.log 2>&1 & echo restarted \$!"
    fi
  fi
  sleep 300
done
