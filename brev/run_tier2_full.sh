#!/usr/bin/env bash
# The full-budget 30-clip Tier 2 run, several clips per box, strictly sequential.
#
# Packing clips onto a box is a cost decision, not a scientific one: provisioning plus the image
# build is ~13 min of billed time that 30 separate boxes would pay 30 times. The clips run one at a
# time so each gets the whole card and the timing stays comparable with the AB350 cells.
#
# Everything about a clip is identical to brev/run_ab350.sh -- seed 42, 349,999,104 steps, a
# checkpoint every 49,999,872, every checkpoint evaluated at world_count=1, and scene.support read
# from that clip's own data. Three things are parameters, because the combined-recipe run changes
# them and nothing else:
#
#   ARM_FLAGS    recipe changes, applied once per box before the first clip (brev/arm_flags.sh).
#   ASSET_ROOT   where the references are mounted in the container.
#   RESULTS_DIR  where the per-clip results go, so one box can hold two runs without mixing them.
#
# The flags are applied ONCE per box, not per clip: the patches are idempotent, but a patch that
# fails has to fail before the first 4.4 h of training, not between clips two and three.
set -uo pipefail

MAP="${1:?episode map: lines of '<episode> <sequence_id>'}"
STEPS="${2:-349999104}"
SAVE_EVERY="${3:-49999872}"        # 2048 * 24414
ASSETS="${ASSET_ROOT:-/v2d/assets/t2_raw}"
OUT="${RESULTS_DIR:-$HOME/t2_results}"
# Seed 42 for every run so far. A second seed is a separate arm with its own box prefix and its
# own results tree, so the only thing that has to change on the box is this.
SEED="${V2D_SEED:-42}"
ARM_FLAGS="${ARM_FLAGS:-}"

for n in "$STEPS" "$SAVE_EVERY"; do
  (( n % 2048 == 0 )) || { echo "$n is not a multiple of world_count 2048" >&2; exit 2; }
done
(( STEPS >= 409600 )) || { echo "below the 409,600-step replay minimum" >&2; exit 2; }
[ -f "$MAP" ] || { echo "no episode map at $MAP" >&2; exit 2; }

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

# shellcheck source=brev/support_flag.sh
. "$(dirname "${BASH_SOURCE[0]}")/support_flag.sh"
# shellcheck source=brev/arm_flags.sh
. "$(dirname "${BASH_SOURCE[0]}")/arm_flags.sh"

# Patch the container for this box's recipe before anything is trained, and record what the box
# actually carries next to its results. A box whose patches did not apply must not train at all.
ARM_OVERRIDES="$(arm_flags_apply "$ARM_FLAGS" "box $(hostname)")" || exit 1
printf '%s\n' "${ARM_FLAGS:-(stock)}" > "$OUT/arm_flags.txt"
echo "arm_flags=${ARM_FLAGS:-(stock)} overrides=${ARM_OVERRIDES:-(none)} \
object_orientation_std=${V2D_OBJECT_ORIENTATION_STD:-(patch default)} assets=$ASSETS" \
  | tee -a "$OUT/summary.txt"

TOTAL=$(grep -c '[^[:space:]]' "$MAP")
INDEX=0
while read -r EP SEQ <&3; do
  [ -n "${EP:-}" ] || continue
  INDEX=$((INDEX + 1))
  TAG="e$(printf '%02d' "$EP")"
  CLIP="$OUT/$TAG"
  PARQUET="$ASSETS/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"
  RUN_DIR="outputs/t2_${TAG}"

  if [ -f "$CLIP/DONE" ]; then
    echo "[$INDEX/$TOTAL] $TAG already done; skipping" | tee -a "$OUT/summary.txt"
    continue
  fi
  mkdir -p "$CLIP"

  say "[$INDEX/$TOTAL] $TAG $SEQ"
  if [ "${DRY_RUN:-0}" != 1 ] && ! run "test -f '$PARQUET'"; then
    echo "$TAG MISSING REFERENCE $PARQUET" | tee -a "$OUT/summary.txt"; continue
  fi
  run "md5sum '$PARQUET'" | tr -d '\r' | tee "$CLIP/reference.md5"
  SUPPORT="$(support_flag_or_die "$PARQUET" "$TAG")" || {
    echo "$TAG SUPPORT UNRESOLVED -- skipped" | tee -a "$OUT/summary.txt"; continue; }
  echo "$TAG $SUPPORT" | tee -a "$CLIP/summary.txt" | tee -a "$OUT/summary.txt"

  say "[$INDEX/$TOTAL] $TAG: train seed $SEED, $STEPS steps (checkpoint every $SAVE_EVERY)"
  START=$(date +%s)
  # train.log per clip, so the launcher's stall probe has one file per clip to watch grow.
  # W&B runs offline on the box (no key, no network) into /workspace/$RUN_DIR/wandb; the run dir is
  # copied next to the checkpoints below and synced from a laptop with `wandb sync -e <entity>`.
  # Checkpoint upload stays off so the offline dir does not carry a second copy of every policy.
  run "cd /workspace && WANDB_MODE=offline python scripts/train_flash_sac.py \
    experiment=g1_recon_body_flash_sac \"task.parquet='${PARQUET}'\" \
    training.total_environment_steps=${STEPS} \
    training.checkpoint.save_interval_environment_steps=${SAVE_EVERY} \
    $SUPPORT $ARM_OVERRIDES logging.mode=offline logging.upload_checkpoints=false \
    logging.project=v2d-t2-combined logging.run_name=t2_${TAG}_s${SEED} \
    reset.seed=${SEED} output_dir=$RUN_DIR" 2>&1 | tee "$CLIP/train.log" | tail -20
  echo "$TAG train seconds=$(( $(date +%s) - START ))" | tee -a "$CLIP/summary.txt"
  if run "test -d /workspace/$RUN_DIR/wandb"; then
    docker exec -i "$CID" tar -C "/workspace/$RUN_DIR" -cf - wandb < /dev/null | tar -C "$CLIP" -xf - \
      && echo "$TAG wandb offline dir: $(du -sh "$CLIP/wandb" | cut -f1)" | tee -a "$CLIP/summary.txt" \
      || echo "$TAG wandb offline dir NOT copied" | tee -a "$CLIP/summary.txt"
  fi

  CKPTS="$(run "ls -1 /workspace/$RUN_DIR/policy_*.safetensors 2>/dev/null" | tr -d '\r')"
  [ "${DRY_RUN:-0}" = 1 ] && CKPTS="/workspace/$RUN_DIR/policy_${STEPS}.safetensors"
  if [ -z "$CKPTS" ]; then
    echo "$TAG produced no checkpoint" | tee -a "$CLIP/summary.txt" | tee -a "$OUT/summary.txt"
    touch "$CLIP/DONE"; continue
  fi

  for CKPT in $CKPTS; do
    CTAG="$(basename "$CKPT" .safetensors)"
    say "[$INDEX/$TOTAL] $TAG: evaluate $CTAG (world_count=1, the submission world count)"
    run "cd /workspace && WANDB_MODE=disabled python scripts/evaluate_policy.py \
      \"evaluation.checkpoint=${CKPT}\" evaluation.world_count=1 \
      \"evaluation.metrics_output=${CKPT}.metrics.json\" \
      \"evaluation.object_trajectories_output=${CKPT}.objects.parquet\"" 2>&1 \
      | tee "$CLIP/eval_${CTAG}.log" | tail -8
    mkdir -p "$CLIP/$CTAG"
    if [ "${DRY_RUN:-0}" != 1 ]; then
      # `docker exec cat`, never `docker cp`: cp fails on this box's ssh-agent mount and once
      # silently produced an empty parquet that was not noticed until scoring.
      docker exec -i "$CID" cat "${CKPT}.metrics.json"    < /dev/null > "$CLIP/$CTAG/metrics.json"    2>/dev/null
      docker exec -i "$CID" cat "${CKPT}.objects.parquet" < /dev/null > "$CLIP/$CTAG/objects.parquet" 2>/dev/null
      docker exec -i "$CID" cat "$CKPT"                   < /dev/null > "$CLIP/$CTAG/policy.safetensors" 2>/dev/null
    fi
    [ -s "$CLIP/$CTAG/objects.parquet" ] \
      && echo "$TAG $CTAG exported $(stat -c%s "$CLIP/$CTAG/objects.parquet") bytes" | tee -a "$CLIP/summary.txt" \
      || { rm -f "$CLIP/$CTAG/objects.parquet"; echo "$TAG $CTAG no trajectory" | tee -a "$CLIP/summary.txt"; }
  done

  # The run directory carries the replay buffer, ~165 MB a clip; four clips would crowd the disk.
  run "rm -rf /workspace/$RUN_DIR" >/dev/null 2>&1
  touch "$CLIP/DONE"
  echo "$TAG done, $(find "$CLIP" -name objects.parquet | wc -l) trajectories" | tee -a "$OUT/summary.txt"
done 3< "$MAP"

DONE_CLIPS=$(find "$OUT" -maxdepth 2 -name DONE | wc -l)
echo "finished $DONE_CLIPS of $TOTAL clips" | tee -a "$OUT/summary.txt"
touch "$OUT/DONE"
say "Done: $OUT"
