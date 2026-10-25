#!/usr/bin/env bash
# Tier 1 fallback entry: one short-budget policy per Tier 1 clip, evaluated and exported.
#
# The point of this run is a VALID submission, not a good score. 409,600 steps is the smallest
# legal budget (the replay minimum; it must also divide 2048), so this is deliberately a weak
# policy -- B1 scored AUC 0.1303 at this budget. What it buys is a real entry: a real checkpoint,
# a real evaluator run, and the three provenance hashes the kit demands, none of which exist for
# the kinematic retarget.
#
# world_count=1 on purpose, and it is a correctness choice rather than a cost one. The kit's
# config sets worlds_per_episode=1, and pack_submission.py rejects an episode outright if ANY
# world in the submitted trajectory carries environment_valid=false ("the evaluator sanitised a
# non-finite pose there"). Evaluating 4096 worlds and submitting world 0 would therefore let any
# one of 4095 unsubmitted worlds veto the episode -- and at this budget non-finite poses are
# likely. One world is exactly what the submission carries.
set -uo pipefail

STEPS="${1:-409600}"
MAP="${2:-$HOME/tier1/tier1_episode_map.txt}"
ASSETS_IN_CONTAINER=/v2d/assets/tier1
OUT="$HOME/tier1_results"
ROLL="$OUT/rollouts"

(( STEPS % 2048 == 0 )) || { echo "steps must be a multiple of 2048, got $STEPS" >&2; exit 2; }
(( STEPS >= 409600 )) || { echo "steps must reach the 409,600-step replay minimum" >&2; exit 2; }
[ -f "$MAP" ] || { echo "no episode map at $MAP" >&2; exit 2; }

mkdir -p "$ROLL"
rm -f "$OUT/DONE"
say() { printf '\n== %s (%s)\n' "$*" "$(date -u +%H:%M:%S)"; }

if [ "${DRY_RUN:-0}" = 1 ]; then
  CID=DRYRUN
else
  CID="$(docker ps -q --filter name=flash_chord | head -1)"
  [ -n "$CID" ] || { echo "no flash_chord container is running" >&2; exit 1; }
fi
run() {
  if [ "${DRY_RUN:-0}" = 1 ]; then printf 'container$ %s\n' "$1"; return 0; fi
  docker exec -i "$CID" bash -lc "$1" < /dev/null
}

TOTAL=$(wc -l < "$MAP")
INDEX=0
while read -r EPISODE SEQUENCE <&3; do
  [ -n "${EPISODE:-}" ] || continue
  INDEX=$((INDEX + 1))
  PARQUET="$ASSETS_IN_CONTAINER/whole_body/soma/sequence_id=${SEQUENCE}/robot_name=g1/data.parquet"
  TAG="$(printf 'e%02d' "$EPISODE")"
  RUN_DIR="outputs/tier1_${TAG}"

  if [ -s "$ROLL/$(printf 'episode_%06d.parquet' "$EPISODE")" ]; then
    echo "$TAG already exported; skipping" | tee -a "$OUT/summary.txt"
    continue
  fi

  say "[$INDEX/$TOTAL] $TAG $SEQUENCE: train $STEPS steps"
  if [ "${DRY_RUN:-0}" != 1 ] && ! run "test -f '$PARQUET'"; then
    echo "$TAG MISSING REFERENCE $PARQUET" | tee -a "$OUT/summary.txt"
    continue
  fi
  START=$(date +%s)
  run "cd /workspace && WANDB_MODE=disabled python scripts/train_flash_sac.py \
    experiment=g1_recon_body_flash_sac \"task.parquet='${PARQUET}'\" \
    training.total_environment_steps=${STEPS} \
    training.checkpoint.save_interval_environment_steps=${STEPS} \
    scene.support=false logging.mode=disabled \
    reset.seed=42 output_dir=$RUN_DIR" 2>&1 | tee "$OUT/train_${TAG}.log" | tail -6
  echo "$TAG train seconds=$(( $(date +%s) - START ))" | tee -a "$OUT/summary.txt"

  CHECKPOINT="$(run "ls -1 /workspace/$RUN_DIR/policy_*.safetensors 2>/dev/null | tail -1" | tr -d '\r')"
  [ "${DRY_RUN:-0}" = 1 ] && CHECKPOINT="/workspace/$RUN_DIR/policy_${STEPS}.safetensors"
  if [ -z "$CHECKPOINT" ]; then
    echo "$TAG produced no checkpoint" | tee -a "$OUT/summary.txt"
    continue
  fi

  say "[$INDEX/$TOTAL] $TAG: evaluate (world_count=1)"
  run "cd /workspace && WANDB_MODE=disabled python scripts/evaluate_policy.py \
    \"evaluation.checkpoint=${CHECKPOINT}\" evaluation.world_count=1 \
    \"evaluation.metrics_output=${CHECKPOINT}.metrics.json\" \
    \"evaluation.object_trajectories_output=${CHECKPOINT}.objects.parquet\"" 2>&1 \
    | tee "$OUT/eval_${TAG}.log" | tail -6

  # `docker exec cat`, never `docker cp`: cp fails on this box's ssh-agent mount and once
  # silently produced an empty parquet that was not noticed until scoring.
  DEST="$ROLL/$(printf 'episode_%06d.parquet' "$EPISODE")"
  docker exec -i "$CID" cat "${CHECKPOINT}.objects.parquet" < /dev/null > "$DEST" 2>/dev/null
  docker exec -i "$CID" cat "${CHECKPOINT}.metrics.json" < /dev/null > "$OUT/metrics_${TAG}.json" 2>/dev/null
  if [ -s "$DEST" ]; then
    echo "$TAG exported $(stat -c%s "$DEST") bytes" | tee -a "$OUT/summary.txt"
  else
    rm -f "$DEST"; echo "$TAG NO TRAJECTORY PARQUET" | tee -a "$OUT/summary.txt"
  fi
  # The policy and its replay state are ~165 MB per clip and 30 clips would fill the disk.
  run "rm -rf /workspace/$RUN_DIR" >/dev/null 2>&1
done 3< "$MAP"

EXPORTED=$(ls -1 "$ROLL" 2>/dev/null | grep -c '^episode_[0-9]*\.parquet$' || true)
echo "exported $EXPORTED of $TOTAL episodes" | tee -a "$OUT/summary.txt"
touch "$OUT/DONE"
say "Done: $OUT"
