#!/usr/bin/env bash
# One full-length policy: arm A (stock recipe), e24, gated-cleaned Tier 2 reference.
#
# Every number we have for Tier 2 so far comes from 10M-step runs, which is 2.9% of the
# published recipe. At that budget success rates are near zero and AUC sits around 0.20 for
# every arm, so we have been comparing arms in a regime where none of them works. This run is
# the first look at what the recipe actually does: 349,999,104 steps, the published figure.
#
# Checkpoints every ~50M steps rather than only at the end. At 23k env steps/s this is a 4.2 h
# run, longer than any single thing we have asked of a Brev box, and a save_interval equal to
# the budget would mean a drop at 4.1 h leaves nothing at all.
set -uo pipefail

STEPS="${1:-349999104}"
SAVE_EVERY="${2:-49999872}"     # 2048 * 24414
E24=2026-05-29_15-43-20_watering_can_pour_right_02
PARQUET="${3:-/v2d/assets/gated/whole_body/soma/sequence_id=${E24}/robot_name=g1/data.parquet}"
OUT="$HOME/full350_results"
RUN_DIR="outputs/full350_A_s42"

for n in "$STEPS" "$SAVE_EVERY"; do
  (( n % 2048 == 0 )) || { echo "$n is not a multiple of world_count 2048" >&2; exit 2; }
done
(( STEPS >= 409600 )) || { echo "below the 409,600-step replay minimum" >&2; exit 2; }

mkdir -p "$OUT"; rm -f "$OUT/DONE"
say() { printf '\n== %s (%s)\n' "$*" "$(date -u +%H:%M:%S)"; }

if [ "${DRY_RUN:-0}" = 1 ]; then CID=DRYRUN; else
  CID="$(docker ps -q --filter name=flash_chord | head -1)"
  [ -n "$CID" ] || { echo "no flash_chord container is running" >&2; exit 1; }
fi
run() {
  if [ "${DRY_RUN:-0}" = 1 ]; then printf 'container$ %s\n' "$1"; return 0; fi
  docker exec -i "$CID" bash -lc "$1"
}

if [ "${DRY_RUN:-0}" != 1 ] && ! run "test -f '$PARQUET'"; then
  echo "FATAL: reference not visible in the container at $PARQUET" >&2; exit 1
fi

say "Train arm A, seed 42, $STEPS steps (checkpoint every $SAVE_EVERY)"
START=$(date +%s)
run "cd /workspace && WANDB_MODE=disabled python scripts/train_flash_sac.py \
  experiment=g1_recon_body_flash_sac \"task.parquet='${PARQUET}'\" \
  training.total_environment_steps=${STEPS} \
  training.checkpoint.save_interval_environment_steps=${SAVE_EVERY} \
  scene.support=false logging.mode=disabled \
  reset.seed=42 output_dir=$RUN_DIR" 2>&1 | tee "$OUT/train.log" | tail -30
ELAPSED=$(( $(date +%s) - START ))
echo "train seconds=$ELAPSED" | tee -a "$OUT/summary.txt"

# Evaluate every checkpoint that survived, not just the last: the learning curve in success-rate
# terms is the thing this run exists to produce, and the intermediate ones are already paid for.
CKPTS="$(run "ls -1 /workspace/$RUN_DIR/policy_*.safetensors 2>/dev/null" | tr -d '\r')"
[ "${DRY_RUN:-0}" = 1 ] && CKPTS="/workspace/$RUN_DIR/policy_${STEPS}.safetensors"
[ -n "$CKPTS" ] || { echo "no checkpoints at all" | tee -a "$OUT/summary.txt"; touch "$OUT/DONE"; exit 1; }

for CKPT in $CKPTS; do
  TAG="$(basename "$CKPT" .safetensors)"
  say "Evaluate $TAG (world_count=1, the submission world count)"
  run "cd /workspace && WANDB_MODE=disabled python scripts/evaluate_policy.py \
    \"evaluation.checkpoint=${CKPT}\" evaluation.world_count=1 \
    \"evaluation.metrics_output=${CKPT}.metrics.json\" \
    \"evaluation.object_trajectories_output=${CKPT}.objects.parquet\"" 2>&1 \
    | tee "$OUT/eval_${TAG}.log" | tail -8
  mkdir -p "$OUT/$TAG"
  docker exec -i "$CID" cat "${CKPT}.metrics.json"   > "$OUT/$TAG/metrics.json"    2>/dev/null
  docker exec -i "$CID" cat "${CKPT}.objects.parquet" > "$OUT/$TAG/objects.parquet" 2>/dev/null
  [ -s "$OUT/$TAG/objects.parquet" ] \
    && echo "$TAG exported $(stat -c%s "$OUT/$TAG/objects.parquet") bytes" | tee -a "$OUT/summary.txt" \
    || { rm -f "$OUT/$TAG/objects.parquet"; echo "$TAG no trajectory" | tee -a "$OUT/summary.txt"; }
done

touch "$OUT/DONE"
say "Done: $OUT"
