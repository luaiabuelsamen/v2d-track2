#!/usr/bin/env bash
# Stage a warm-start fine-tune: the shipped actors, plus a NEW reference to fine-tune them onto.
#
# Two targets, and they differ in how much work precedes staging:
#
#   --target tier1   the clean Tier 1 references, already retargeted on the Jetson
#                    ($S/t1_clean/motion/tier1, 30/30 at 0.0 mm rigid-fit residual). Ready now.
#
#   --target tier3   the Tier 3 reconstruction. NOT ready from the reconstruction alone: what
#                    FoundationPose produces is an object TRAJECTORY, and the trainer wants a
#                    retargeted whole-body motion parquet solved against that trajectory. The
#                    path is `v2d retarget --object-npz <recon dir>`, which is exactly what the
#                    --object-npz flag exists for, and it has to run (free, CPU, ~4 min/clip)
#                    before this script has anything to stage. This script refuses rather than
#                    staging Tier 2 motion and calling it Tier 3.
#
# Layout is the same trick the eval staging uses: each shipped actor records its own
# `/v2d/assets/<stage>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet`, so the new
# reference goes at that same path and the actor is handed to the runner by
# `training.warm_start_actor_path`. The warm start does NOT validate the reference contract -- that
# is the whole point of brev/patch_warm_start.py -- so nothing has to be spoofed.
#
#   brev/stage_warm_start.sh --target tier1 [--episodes 3,9,18] [--out DIR]
set -uo pipefail

REPO=/home/jetson3/projects/v2d-challenge
S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
TARGET=""
EPISODES=""
OUT=""
RECON="${RECON:-$S/t3/motion/tier3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --recon) RECON="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$TARGET" in
  tier1)
    MOTION="$S/t1_clean/motion/tier1"
    OUT="${OUT:-$S/ws_tier1/assets}"
    ;;
  tier3)
    MOTION="$RECON"
    OUT="${OUT:-$S/ws_tier3/assets}"
    if [ ! -d "$MOTION" ]; then
      cat >&2 <<EOF
FATAL: no retargeted Tier 3 motion tree at $MOTION

The reconstruction gives object poses; the trainer needs a whole-body solve against them. Run that
first, free, on the Jetson -- 30 clips, about 4 min each:

  v2d retarget --dataset-root \$S/data --repo /home/jetson3/projects/v2d-upstream \\
      --soma-root <SOMA_neutral dir> --tiers tier2 --object-npz <recon episode_%06d.npz dir> \\
      --seq-root \$S/t3/seq --out-root \$S/t3/motion --log-dir \$S/t3/logs \\
      --validation-dir \$S/t3/validate --workers 1 -o \$S/t3/retarget.json

One worker, not two: two concurrent workers were OOM-killed on this 15 GB machine (e05, e14).
Refusing rather than staging Tier 2 motion under a Tier 3 name.
EOF
      exit 3
    fi
    ;;
  *) echo "usage: --target tier1|tier3" >&2; exit 2 ;;
esac

[ -d "$MOTION" ] || { echo "FATAL: no motion tree at $MOTION" >&2; exit 3; }

# The eval stager already does exactly this job -- read each actor's recorded path out of its
# safetensors metadata, put the reference there, copy the support USDA where the scene builder
# looks, rebase the mesh paths, write the manifest last. Reuse it rather than growing a second
# copy that can drift.
ARGS=(--motion "$MOTION" --out "$OUT")
[ -n "$EPISODES" ] && ARGS+=(--episodes "$EPISODES")
bash "$REPO/brev/stage_t1_eval.sh" "${ARGS[@]}" || exit 3

# The warm start needs the actor by path INSIDE the container. The eval staging already copies
# each one to <out>/ckpt/eNN/policy.safetensors, so record the in-container path per clip for the
# runner to pass as training.warm_start_actor_path.
PLAN="${OUT%/}.sources.txt"
WS_PLAN="${OUT%/}.warmstart.txt"
: > "$WS_PLAN"
while read -r EP TAG STAGE SEQ SUPPORT; do
  [ -n "${EP:-}" ] || continue
  printf '%s %s %s %s %s %s\n' "$EP" "$TAG" "$STAGE" "$SEQ" "$SUPPORT" \
    "/v2d/assets/ckpt/$TAG/policy.safetensors" >> "$WS_PLAN"
done < "$PLAN"

echo
echo "warm-start plan: $WS_PLAN ($(grep -c . "$WS_PLAN") clips)"
echo "  reference:  $MOTION"
echo "  actors:     $OUT/ckpt/eNN/policy.safetensors -> /v2d/assets/ckpt/eNN/policy.safetensors"
echo "  runner override: training.warm_start_actor_path=<that path>"
echo "                   training.total_environment_steps=349999104"
echo "                   (warm_start_environment_steps defaults to 299,999,232 => ~50M of training)"
