#!/usr/bin/env bash
# On the box: fine-tune each shipped actor for ~50M steps on a NEW reference, then evaluate it.
#
# One clip at a time, sequentially, because training saturates the card -- unlike the Tier 3
# reconstruction, where the slow stage was I/O-bound and three workers fit.
#
# Per clip: train from the actor with `training.warm_start_actor_path`, which exists because of
# brev/patch_warm_start.py (upstream's resume needs a training_state checkpoint we do not have,
# and refuses a changed reference anyway -- see docs/WARM_START.md). The step counter starts at
# the patch's default 299,999,232, so a 349,999,104 budget is 49,999,872 steps of fine-tuning and
# the save interval writes exactly one new checkpoint. Then evaluate that checkpoint at
# world_count=1 and export the rollout, which is what the kit scorer reads on the Jetson.
#
# PLAN lines: "<ep> <tag> <stage> <seq> <support> <in-container actor path>"
#
#   PLAN=~/ws_plan.txt ARM_FLAGS=start_curriculum,orientation_reward,object_position \
#     V2D_OBJECT_POSITION_STD=0.08 DEADLINE_H=3 bash ~/run_warm_start.sh
set -uo pipefail

PLAN="${PLAN:-$HOME/ws_plan.txt}"
OUT="${OUT:-$HOME/ws_results}"
STEPS="${STEPS:-349999104}"
SAVE_EVERY="${SAVE_EVERY:-49999872}"
SEED="${SEED:-42}"
DEADLINE_H="${DEADLINE_H:-3.0}"
# warm_start is appended here rather than asked for, so a caller cannot request this runner and
# forget the one patch that makes it a fine-tune instead of a 50M-step run from scratch.
ARM_FLAGS="${ARM_FLAGS:-start_curriculum,orientation_reward,object_position},warm_start"
START=$(date +%s)
mkdir -p "$OUT/rollouts"

say() { printf '\n== %s (%s)\n' "$*" "$(date -u +%H:%M:%S)"; }
container_id() { docker ps -q --filter name=flash_chord | head -1; }
# `< /dev/null` is not optional: without it `docker exec -i` inherits the loop's stdin, eats the
# plan, and the loop ends after one clip having written every success marker.
run() {
  local cid
  cid="$(container_id)" || return 1
  [ -n "$cid" ] || { echo "no flash_chord container is running" >&2; return 1; }
  docker exec -i "$cid" bash -lc "$1" < /dev/null
}

CID="$(container_id)"
[ -n "$CID" ] || { echo "FATAL: no container; run gate_on_instance.sh build first" >&2; exit 2; }
[ -f "$PLAN" ] || { echo "FATAL: no plan at $PLAN" >&2; exit 2; }
run 'test -d /v2d/assets' || { echo "FATAL: /v2d/assets is not mounted" >&2; exit 2; }

say "apply the recipe: $ARM_FLAGS"
# shellcheck source=brev/arm_flags.sh
. "$HOME/arm_flags.sh"
ARM_OVERRIDES="$(arm_flags_apply "$ARM_FLAGS" "warm-start $(hostname)")" || exit 1
# Prove the warm-start keys exist before training anything, the same way the eval box proves the
# objective fields: a missing key would otherwise train 50M steps from scratch and look fine.
run 'python -c "
import dataclasses
from flash_chord.training.flash_sac.config import TrainingConfig, OptimizationConfig
t={f.name for f in dataclasses.fields(TrainingConfig)}
o={f.name for f in dataclasses.fields(OptimizationConfig)}
missing=({\"warm_start_actor_path\",\"warm_start_environment_steps\"}-t) | ({\"warm_start_actor_rate_multiplier\",\"warm_start_guard_updates\"}-o)
raise SystemExit(f\"config is missing {missing}\" if missing else 0)
"' || { echo "FATAL: the warm-start patch did not take; refusing to train" >&2; exit 2; }
echo "warm-start keys present; overrides: ${ARM_OVERRIDES:-(none)}"

TOTAL="$(grep -c . "$PLAN")"
INDEX=0
: > "$OUT/summary.txt"
while read -r EP TAG STAGE SEQ SUPPORT ACTOR <&3; do
  [ -n "${EP:-}" ] || continue
  INDEX=$((INDEX + 1))
  DEST="$OUT/rollouts/$(printf 'episode_%06d.parquet' "$EP")"
  [ -s "$DEST" ] && { echo "$TAG already done; skipping" | tee -a "$OUT/summary.txt"; continue; }
  if awk -v e="$(awk -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.3f", (n-s)/3600}')" \
         -v d="$DEADLINE_H" 'BEGIN{exit !(e >= d)}'; then
    echo "DEADLINE ${DEADLINE_H} h reached after $((INDEX - 1)) clips" | tee -a "$OUT/summary.txt"
    break
  fi

  PARQUET="/v2d/assets/$STAGE/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"
  run "test -f '$PARQUET'" || { echo "$TAG MISSING REFERENCE" | tee -a "$OUT/summary.txt"; continue; }
  run "test -f '$ACTOR'" || { echo "$TAG MISSING ACTOR $ACTOR" | tee -a "$OUT/summary.txt"; continue; }
  # scene.support is carried from what the actor TRAINED with, so the scene is not a second variable.
  SUPPORT_FLAG="scene.support=$([ "$SUPPORT" = True ] && echo true || echo false)"
  RUN_DIR="outputs/ws_${TAG}"

  say "[$INDEX/$TOTAL] $TAG: fine-tune from $(basename "$(dirname "$ACTOR")") ($SUPPORT_FLAG)"
  T0=$(date +%s)
  run "cd /workspace && WANDB_MODE=offline python scripts/train_flash_sac.py \
    experiment=g1_recon_body_flash_sac \"task.parquet='${PARQUET}'\" \
    training.warm_start_actor_path='${ACTOR}' \
    training.total_environment_steps=${STEPS} \
    training.checkpoint.save_interval_environment_steps=${SAVE_EVERY} \
    $SUPPORT_FLAG $ARM_OVERRIDES logging.mode=offline logging.upload_checkpoints=false \
    logging.project=v2d-warm-start logging.run_name=ws_${TAG} \
    reset.seed=${SEED} output_dir=$RUN_DIR" 2>&1 | tee "$OUT/train_${TAG}.log" | tail -12
  echo "$TAG train seconds=$(( $(date +%s) - T0 ))" | tee -a "$OUT/summary.txt"
  # The line the patch prints. Its absence means the actor was NOT loaded and this is a 50M-step
  # run from scratch wearing a fine-tune's name, so it is checked rather than assumed.
  if ! grep -q "warm start: actor from" "$OUT/train_${TAG}.log"; then
    echo "$TAG NO WARM START LINE -- treating as a failed fine-tune" | tee -a "$OUT/summary.txt"
    continue
  fi
  grep -m1 "warm start: actor from" "$OUT/train_${TAG}.log" | tee -a "$OUT/summary.txt"

  CKPT="$(run "ls -1 /workspace/$RUN_DIR/policy_*.safetensors 2>/dev/null | tail -1" | tr -d '\r')"
  [ -n "$CKPT" ] || { echo "$TAG produced no checkpoint" | tee -a "$OUT/summary.txt"; continue; }

  say "[$INDEX/$TOTAL] $TAG: evaluate the fine-tuned actor (world_count=1)"
  run "cd /workspace && WANDB_MODE=disabled python scripts/evaluate_policy.py \
    \"evaluation.checkpoint=${CKPT}\" evaluation.world_count=1 \
    \"evaluation.metrics_output=${CKPT}.metrics.json\" \
    \"evaluation.object_trajectories_output=${CKPT}.objects.parquet\" \
    logging.mode=disabled" 2>&1 | tee "$OUT/eval_${TAG}.log" | tail -4

  docker exec -i "$CID" cat "${CKPT}.objects.parquet" < /dev/null > "$DEST" 2>/dev/null
  docker exec -i "$CID" cat "${CKPT}.metrics.json" < /dev/null \
    > "$OUT/rollouts/$(printf 'episode_%06d.json' "$EP")" 2>/dev/null
  docker exec -i "$CID" cat "$CKPT" < /dev/null > "$OUT/rollouts/${TAG}_policy.safetensors" 2>/dev/null
  if [ -s "$DEST" ]; then
    echo "$TAG exported $(stat -c%s "$DEST") bytes" | tee -a "$OUT/summary.txt"
  else
    rm -f "$DEST"
    echo "$TAG EXPORT EMPTY -- removed" | tee -a "$OUT/summary.txt"
  fi
done 3< "$PLAN"

EXPORTED="$(find "$OUT/rollouts" -name 'episode_*.parquet' | wc -l)"
echo "exported $EXPORTED of $TOTAL clips" | tee -a "$OUT/summary.txt"
echo "$EXPORTED" > "$OUT/COUNT"
if [ "$EXPORTED" = "$TOTAL" ]; then
  touch "$OUT/DONE"
else
  echo "FATAL: exported $EXPORTED of $TOTAL -- not reporting this run as finished" >&2
  exit 4
fi
