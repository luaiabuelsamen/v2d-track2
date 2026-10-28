#!/usr/bin/env bash
# On the box: evaluate each wave-2 checkpoint against the CLEAN Tier 1 trajectory. No training.
#
# The container is already up (gate_on_instance.sh build) and ~/v2d_assets holds the staged tree,
# so `/v2d/assets/<stage>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet` is exactly
# the path each checkpoint recorded at training time -- but now with Tier 1's clean object track in
# it. `evaluate_policy.py` rebuilds everything else from the checkpoint's own saved config, so
# world count, reset mode, scene.support and the object are held at their trained values and the
# trajectory is the single variable.
#
# world_count=1 on purpose, as in run_tier1_backup.sh: the kit's config sets worlds_per_episode=1
# and pack_submission.py rejects an episode if ANY submitted world carries environment_valid=false,
# so evaluating many worlds and submitting world 0 would let an unsubmitted world veto the clip.
#
#   PLAN=~/t1_plan.txt DEADLINE_H=2.5 bash ~/t1_eval_box.sh
set -uo pipefail

PLAN="${PLAN:-$HOME/t1_plan.txt}"
OUT="$HOME/t1_eval_results"
DEADLINE_H="${DEADLINE_H:-2.5}"
START=$(date +%s)
mkdir -p "$OUT/rollouts"

say() { printf '\n== %s (%s)\n' "$*" "$(date -u +%H:%M:%S)"; }
container_id() { docker ps -q --filter name=flash_chord | head -1; }
# -i and never -it: the repo's wrapper uses `docker exec -it`, and -t fails on this box with
# "the input device is not a TTY", which silently lost every container call once already.
run() {
  local cid
  cid="$(container_id)" || return 1
  [ -n "$cid" ] || { echo "no flash_chord container is running" >&2; return 1; }
  docker exec -i "$cid" bash -lc "$1"
}

CID="$(container_id)"
[ -n "$CID" ] || { echo "FATAL: no container; run gate_on_instance.sh build first" >&2; exit 2; }
[ -f "$PLAN" ] || { echo "FATAL: no plan at $PLAN" >&2; exit 2; }

# The staged tree must be visible in the container before anything else, or every clip fails the
# same way 30 times over.
run 'test -d /v2d/assets' || { echo "FATAL: /v2d/assets is not mounted in the container" >&2; exit 2; }

TOTAL="$(grep -c . "$PLAN")"
INDEX=0
: > "$OUT/summary.txt"
while read -r EP TAG STAGE SEQ SUPPORT; do
  [ -n "${EP:-}" ] || continue
  INDEX=$((INDEX + 1))
  ELAPSED_H="$(awk -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.3f", (n-s)/3600}')"
  if awk -v e="$ELAPSED_H" -v d="$DEADLINE_H" 'BEGIN{exit !(e >= d)}'; then
    echo "DEADLINE ${DEADLINE_H} h reached after $((INDEX - 1)) clips; stopping" | tee -a "$OUT/summary.txt"
    break
  fi

  DEST="$OUT/rollouts/$(printf 'episode_%06d.parquet' "$EP")"
  if [ -s "$DEST" ]; then
    echo "$TAG already exported; skipping" | tee -a "$OUT/summary.txt"
    continue
  fi

  CKPT="/v2d/assets/ckpt/$TAG/policy.safetensors"
  PARQUET="/v2d/assets/$STAGE/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"
  if ! run "test -f '$CKPT'"; then
    echo "$TAG MISSING CHECKPOINT $CKPT" | tee -a "$OUT/summary.txt"; continue
  fi
  if ! run "test -f '$PARQUET'"; then
    echo "$TAG MISSING CLEAN REFERENCE $PARQUET" | tee -a "$OUT/summary.txt"; continue
  fi

  say "[$INDEX/$TOTAL] $TAG $SEQ: evaluate against the clean Tier 1 track (support=$SUPPORT)"
  T0=$(date +%s)
  run "cd /workspace && WANDB_MODE=disabled python scripts/evaluate_policy.py \
    \"evaluation.checkpoint=${CKPT}\" evaluation.world_count=1 \
    \"evaluation.metrics_output=/workspace/outputs/t1_${TAG}.metrics.json\" \
    \"evaluation.object_trajectories_output=/workspace/outputs/t1_${TAG}.objects.parquet\" \
    logging.mode=disabled" 2>&1 | tee "$OUT/eval_${TAG}.log" | tail -4
  echo "$TAG eval seconds=$(( $(date +%s) - T0 ))" | tee -a "$OUT/summary.txt"

  # `docker exec cat`, never `docker cp`: cp fails on this box's ssh-agent mount and once
  # silently produced an empty parquet that went unnoticed until scoring.
  docker exec -i "$CID" cat "/workspace/outputs/t1_${TAG}.objects.parquet" < /dev/null > "$DEST" 2>/dev/null
  docker exec -i "$CID" cat "/workspace/outputs/t1_${TAG}.metrics.json" < /dev/null \
    > "$OUT/rollouts/$(printf 'episode_%06d.json' "$EP")" 2>/dev/null
  if [ -s "$DEST" ]; then
    echo "$TAG exported $(stat -c%s "$DEST") bytes" | tee -a "$OUT/summary.txt"
  else
    rm -f "$DEST"
    echo "$TAG EXPORT EMPTY -- removed so it cannot be mistaken for a result" | tee -a "$OUT/summary.txt"
  fi
done < "$PLAN"

EXPORTED="$(find "$OUT/rollouts" -name 'episode_*.parquet' | wc -l)"
echo "exported $EXPORTED of $TOTAL clips" | tee -a "$OUT/summary.txt"
[ "$EXPORTED" = "$TOTAL" ] && touch "$OUT/DONE"
echo "$EXPORTED" > "$OUT/COUNT"
