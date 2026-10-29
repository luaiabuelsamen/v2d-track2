#!/usr/bin/env bash
# Swap a box onto its pruned plan at a clip boundary.
#
# The runner walks its plan on fd 3, opened once, so rewriting the file mid-run changes nothing --
# the queue has to be swapped by restarting the runner, and the runner skips clips whose rollout
# already exists. Killing mid-clip would throw away real training, so this waits for the clip in
# flight to export, then kills the runner by exact PID (never a command-string match) and restarts
# it on the pruned plan.
set -uo pipefail
S=/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad
BOX=$1; PLAN=$2; WAIT_FOR=$3   # e.g. v2d-t1f-2c  .../plan_pruned.txt  episode_000012
rssh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 "$BOX" "$@" 2>/dev/null | grep -v Pseudo; }

for i in $(seq 1 240); do
  if [ -n "$(rssh "ls ~/t1_eval_results/rollouts/$WAIT_FOR.parquet 2>/dev/null")" ]; then
    echo "$(date -u +%H:%M) $BOX: $WAIT_FOR exported -- pruning"
    # Runner first so it cannot start the next clip while we work, then what it spawned.
    PIDS=$(rssh "ps -eo pid,args --no-headers | awk '\$2==\"bash\" && \$3==\"/home/shadeform/run_warm_start.sh\" {print \$1}' | tr '\n' ' '")
    for P in $PIDS; do rssh "kill $P" && echo "   killed runner $P"; done
    sleep 3
    TP=$(rssh "ps -eo pid,args --no-headers | awk '\$3==\"scripts/train_flash_sac.py\" || (\$2==\"docker\" && \$3==\"exec\") {print \$1}' | tr '\n' ' '")
    for P in $TP; do rssh "kill $P" >/dev/null && echo "   killed worker $P"; done
    sleep 3
    scp -o BatchMode=yes "$PLAN" "$BOX":~/t1_plan.txt >/dev/null || { echo "   SCP FAILED"; exit 1; }
    echo "   queue now: $(rssh "awk '{print \$2}' ~/t1_plan.txt | tr '\n' ' '")"
    rssh "nohup env PLAN=\$HOME/t1_plan.txt OUT=\$HOME/t1_eval_results DEADLINE_H=6.5 \
      ARM_FLAGS=start_curriculum,orientation_reward,object_position \
      V2D_OBJECT_POSITION_STD=0.08 V2D_OBJECT_ORIENTATION_STD=0.3 \
      bash \$HOME/run_warm_start.sh >> \$HOME/t1_eval_results/run.log 2>&1 & echo restarted \$!"
    sleep 120
    echo "   GPU now: $(rssh 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader')"
    rssh "tail -2 ~/t1_eval_results/run.log" | sed 's/^/   /'
    exit 0
  fi
  sleep 60
done
echo "$(date -u +%H:%M) $BOX: gave up waiting for $WAIT_FOR"
