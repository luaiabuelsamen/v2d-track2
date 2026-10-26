#!/usr/bin/env bash
# One cell of the 3-clip full-budget A/B: one arm, one clip, 349,999,104 steps, stock recipe.
#
# The two arms differ in exactly one thing, the parquet, so this script takes the arm name only to
# label the output. Everything else -- recipe, seed, step count, save interval, evaluation world
# count -- is identical across all six cells by construction, because a difference anywhere else
# would be a confound (see docs/AB350_BAR.md).
#
# Evaluates every surviving checkpoint, not just the last: at 350M the learning curve in
# success-rate terms is a result in its own right, and the intermediate checkpoints are paid for.
set -uo pipefail

ARM="${1:?arm: raw or clean}"
EP="${2:?episode index, e.g. 18}"
SEQ="${3:?sequence_id}"
STEPS="${4:-349999104}"
SAVE_EVERY="${5:-49999872}"      # 2048 * 24414

case "$ARM" in raw|clean) ;; *) echo "arm must be raw or clean, not '$ARM'" >&2; exit 2 ;; esac

# The arm name still means the parquet. Recipe changes travel separately, in ARM_FLAGS, so one
# runner covers every arm and nothing about a cell is implied by its name. Validated here, before
# anything is provisioned or trained, because an unknown flag at hour four costs a cell.
# shellcheck source=brev/arm_flags.sh
. "$(dirname "${BASH_SOURCE[0]}")/arm_flags.sh"
ARM_FLAGS="${ARM_FLAGS:-}"
SLUG="$(arm_flags_slug "$ARM_FLAGS")" || exit 2
# The slug is empty for the stock recipe, so stock cells keep the paths they have always had, and a
# flagged cell cannot overwrite the stock cell it is being measured against.
CELL="${ARM}_e$(printf '%02d' "$EP")${SLUG}"

PARQUET="/v2d/assets/ab_${ARM}/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"
OUT="$HOME/ab350_results/${CELL}"
RUN_DIR="outputs/ab350_${CELL}"

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
  docker exec -i "$CID" bash -lc "$1" < /dev/null
}

# The arm IS the parquet, so a missing or misplaced reference would silently run the wrong arm.
if [ "${DRY_RUN:-0}" != 1 ] && ! run "test -f '$PARQUET'"; then
  echo "FATAL: arm $ARM reference not visible in the container at $PARQUET" >&2; exit 1
fi
run "md5sum '$PARQUET'" | tr -d '\r' | tee "$OUT/reference.md5"

# scene.support is a per-clip property of the data, not a global preference. Hardcoding it false
# cost us this very experiment once already; see brev/support_flag.sh.
# shellcheck source=brev/support_flag.sh
. "$(dirname "${BASH_SOURCE[0]}")/support_flag.sh"
SUPPORT="$(support_flag_or_die "$PARQUET" "arm $ARM e$EP")" || exit 1
echo "$SUPPORT" | tee -a "$OUT/summary.txt"

# Patch the container's flash_chord for every flag this arm carries, and collect the hydra
# overrides that go with them. Fatal if any patch is missing or refuses: better no cell than a cell
# whose recipe nobody can name afterwards.
ARM_OVERRIDES="$(arm_flags_apply "$ARM_FLAGS" "arm $ARM e$EP")" || exit 1
printf '%s\n' "${ARM_FLAGS:-(stock)}" > "$OUT/arm_flags.txt"
echo "arm_flags=${ARM_FLAGS:-(stock)} overrides=${ARM_OVERRIDES:-(none)}" | tee -a "$OUT/summary.txt"

say "Train arm $ARM${SLUG} e$EP, seed 42, $STEPS steps (checkpoint every $SAVE_EVERY)"
START=$(date +%s)
run "cd /workspace && WANDB_MODE=disabled python scripts/train_flash_sac.py \
  experiment=g1_recon_body_flash_sac \"task.parquet='${PARQUET}'\" \
  training.total_environment_steps=${STEPS} \
  training.checkpoint.save_interval_environment_steps=${SAVE_EVERY} \
  $SUPPORT $ARM_OVERRIDES logging.mode=disabled \
  reset.seed=42 output_dir=$RUN_DIR" 2>&1 | tee "$OUT/train.log" | tail -20
ELAPSED=$(( $(date +%s) - START ))
echo "train seconds=$ELAPSED" | tee -a "$OUT/summary.txt"

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
  if [ "${DRY_RUN:-0}" != 1 ]; then
    docker exec -i "$CID" cat "${CKPT}.metrics.json"    > "$OUT/$TAG/metrics.json"    2>/dev/null
    docker exec -i "$CID" cat "${CKPT}.objects.parquet" > "$OUT/$TAG/objects.parquet" 2>/dev/null
    # The policy itself, because GPU training is not bit-reproducible at a fixed seed and a
    # 4.8 h run is not something to repeat for want of 4.9 MB.
    docker exec -i "$CID" cat "$CKPT"                   > "$OUT/$TAG/policy.safetensors" 2>/dev/null
  fi
  [ -s "$OUT/$TAG/objects.parquet" ] \
    && echo "$TAG exported $(stat -c%s "$OUT/$TAG/objects.parquet") bytes" | tee -a "$OUT/summary.txt" \
    || { rm -f "$OUT/$TAG/objects.parquet"; echo "$TAG no trajectory" | tee -a "$OUT/summary.txt"; }
done

touch "$OUT/DONE"
say "Done: $OUT"
