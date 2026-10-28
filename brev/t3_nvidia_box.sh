#!/usr/bin/env bash
# On the box: reconstruct the Tier 3 object track for every clip with NVIDIA's own modules.
#
# Tier 3 supplies no trajectory ("bring your own reconstruction"), so this is the input we are
# judged on building. The chain is the single-view path through the challenge repo's own
# reconstruction modules -- the stack that produced this dataset's ground truth:
#
#   1 geocalib   video              -> intrinsics JSON (+gravity). The exo camera is static, so one
#                                      solve serves the clip, and nothing from any tier is read.
#   2 moge       video + intrinsics -> per-frame metric depth. FoundationPose needs depth and an
#                                      RGB camera does not give it.
#   3 dino       video + "<object>" -> bbox JSON. The object's name comes from its released mesh
#                                      directory, which is a Tier 3 asset, not a Tier 1 caption.
#   4 sam2       video + bbox       -> per-frame object masks.
#   5 fpose      all of the above
#                + the released mesh -> 6-DoF object pose per frame, in the CAMERA frame.
#
# Frame-0 alignment into the released world happens off this box, free, on the Jetson: FAQ PR 207
# permits using Tier 1's first frame for exactly that, and it is the only Tier 1 read anywhere in
# the pipeline.
#
# Shape of the run: build every image FIRST and stop on the first failure, before any clip is
# attempted, because a half-built chain fails 30 times in a row and bills for all of them. Each
# stage writes a marker so a re-run after a fix skips what already succeeded -- this box is meant
# to be kept and iterated on, not recreated.
#
#   STAGE=build|weights|clips|all DEADLINE_H=4 bash ~/t3_nvidia_box.sh
set -uo pipefail

STAGE="${STAGE:-all}"
DEADLINE_H="${DEADLINE_H:-4}"
WORK="$HOME/t3"
REPO="$WORK/video_to_data"
RECON="$REPO/reconstruction"
WEIGHTS="$WORK/weights"
OUT="$HOME/t3_out"
MARK="$WORK/marks"
START=$(date +%s)
mkdir -p "$WORK" "$WEIGHTS" "$OUT" "$MARK"

MODULES="geocalib moge grounding_dino sam2 foundation_pose"

say() { printf '\n== %s (%s)\n' "$*" "$(date -u +%H:%M:%S)"; }
elapsed_h() { awk -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.3f", (n-s)/3600}'; }
past_deadline() { awk -v e="$(elapsed_h)" -v d="$DEADLINE_H" 'BEGIN{exit !(e >= d)}'; }
done_mark() { [ -f "$MARK/$1" ]; }
mark() { date -u +%FT%TZ > "$MARK/$1"; }

docker info >/dev/null 2>&1 || { echo "FATAL: docker is not usable on this box" >&2; exit 1; }

# --------------------------------------------------------------------------- the repo
if [ ! -d "$RECON" ]; then
  say "clone the challenge repo (sparse: reconstruction/ only)"
  git clone --filter=blob:none --depth 1 --sparse \
    https://github.com/nvidia-isaac/video_to_data.git "$REPO" || exit 1
  ( cd "$REPO" && git sparse-checkout set reconstruction ) || exit 1
fi
cd "$RECON" || exit 1

# --------------------------------------------------------------------------- 1. images
if [ "$STAGE" = build ] || [ "$STAGE" = all ]; then
  for M in $MODULES; do
    if done_mark "build_$M"; then echo "image $M already built"; continue; fi
    say "build image: $M"
    T0=$(date +%s)
    if ! pip install -q -e "modules/v2d_$M/docker" > "$WORK/pip_$M.log" 2>&1; then
      echo "FATAL: pip install of the $M docker wrapper failed; see $WORK/pip_$M.log" >&2
      tail -5 "$WORK/pip_$M.log" >&2; exit 2
    fi
    if ! python -m "v2d.$M.docker.build" > "$WORK/build_$M.log" 2>&1; then
      echo "FATAL: docker build of $M failed after $(( $(date +%s) - T0 ))s" >&2
      echo "       the box is being KEPT so this can be fixed in place; see $WORK/build_$M.log" >&2
      tail -20 "$WORK/build_$M.log" >&2; exit 3
    fi
    echo "$M built in $(( $(date +%s) - T0 ))s"
    mark "build_$M"
  done
  say "all images built"
fi

# --------------------------------------------------------------------------- 2. weights
if [ "$STAGE" = weights ] || [ "$STAGE" = all ]; then
  for M in $MODULES; do
    if done_mark "weights_$M"; then echo "weights $M already present"; continue; fi
    say "weights: $M"
    # FoundationPose defaults to the NGC/TensorRT weights, which the downloader gates behind
    # --accept_nvidia_model_eula. Accepting a model licence is not mine to do, so this takes the
    # nvlabs_pytorch backend instead: the same two runs (refiner 2023-10-28-18-33-37, scorer
    # 2024-01-11-20-02-45), no EULA flag, and mirrored on Hugging Face if gdown is rate-limited.
    EXTRA=""
    [ "$M" = foundation_pose ] && EXTRA="--backend nvlabs_pytorch"
    if ! python -m "v2d.$M.docker.run_download_weights" --output_dir "$WEIGHTS/$M" $EXTRA \
         > "$WORK/weights_$M.log" 2>&1; then
      echo "FATAL: weights download for $M failed; box KEPT; see $WORK/weights_$M.log" >&2
      tail -15 "$WORK/weights_$M.log" >&2; exit 4
    fi
    du -sh "$WEIGHTS/$M" 2>/dev/null
    mark "weights_$M"
  done
  say "all weights present"
fi

# --------------------------------------------------------------------------- 3. the clips
if [ "$STAGE" = clips ] || [ "$STAGE" = all ]; then
  PLAN="$HOME/t3_plan.txt"   # "<ep> <seq> <object_name>"
  [ -f "$PLAN" ] || { echo "FATAL: no plan at $PLAN" >&2; exit 2; }
  TOTAL="$(grep -c . "$PLAN")"
  INDEX=0
  while read -r EP SEQ OBJ; do
    [ -n "${EP:-}" ] || continue
    INDEX=$((INDEX + 1))
    TAG="$(printf 'e%02d' "$EP")"
    CLIP="$WORK/clips/$TAG"
    POSES="$CLIP/poses"
    if [ -s "$OUT/$(printf 'episode_%06d.npz' "$EP")" ] || done_mark "clip_$TAG"; then
      echo "$TAG already reconstructed; skipping"; continue
    fi
    if past_deadline; then
      echo "DEADLINE ${DEADLINE_H} h reached after $((INDEX - 1)) clips; stopping cleanly"
      break
    fi
    VIDEO="$HOME/t3_videos/$(printf 'episode_%06d.mp4' "$EP")"
    MESH="$HOME/t3_meshes/$OBJ"
    MESH_FILE="$(ls "$MESH"/*.obj "$MESH"/*.glb 2>/dev/null | head -1)"
    for need in "$VIDEO" "$MESH_FILE"; do
      [ -n "$need" ] && [ -f "$need" ] || { echo "$TAG MISSING INPUT ($need)"; continue 2; }
    done
    mkdir -p "$CLIP" "$POSES"

    mkdir -p "$CLIP/calib"
    INTR="$CLIP/calib/intrinsics.json"
    say "[$INDEX/$TOTAL] $TAG $OBJ: geocalib"
    python -m v2d.geocalib.docker.run_video_to_calibration \
      --video_path "$VIDEO" --intrinsics_path "$INTR" \
      --distortion_path "$CLIP/calib/distortion.json" \
      --gravity_path "$CLIP/calib/gravity.json" \
      --calibration_path "$CLIP/calib/calibration.json" \
      --weights_path "$WEIGHTS/geocalib" \
      > "$CLIP/geocalib.log" 2>&1 || { echo "$TAG geocalib FAILED"; tail -5 "$CLIP/geocalib.log"; continue; }
    [ -s "$INTR" ] || { echo "$TAG geocalib wrote no intrinsics at $INTR"; continue; }

    say "[$INDEX/$TOTAL] $TAG: moge depth"
    python -m v2d.moge.docker.run_video_to_depth \
      --video_path "$VIDEO" --depth_folder "$CLIP/depth" --intrinsics_folder "$CLIP/intr" \
      --input_intrinsics_path "$INTR" --weights_path "$WEIGHTS/moge" \
      > "$CLIP/moge.log" 2>&1 || { echo "$TAG moge FAILED"; tail -5 "$CLIP/moge.log"; continue; }

    say "[$INDEX/$TOTAL] $TAG: grounding dino on '$OBJ'"
    PROMPT="$(echo "$OBJ" | tr '_' ' ')"
    python -m v2d.grounding_dino.docker.run_video_to_object_bboxes \
      --video_path "$VIDEO" --prompt "$PROMPT" --output_path "$CLIP/bbox.json" \
      --model_dir "$WEIGHTS/grounding_dino" \
      > "$CLIP/dino.log" 2>&1 || { echo "$TAG dino FAILED"; tail -5 "$CLIP/dino.log"; continue; }

    say "[$INDEX/$TOTAL] $TAG: sam2 masks"
    python -m v2d.sam2.docker.run_video_to_masks \
      --video_path "$VIDEO" --prompts_path "$CLIP/bbox.json" --masks_dir "$CLIP/masks" \
      --weights_dir "$WEIGHTS/sam2" \
      > "$CLIP/sam2.log" 2>&1 || { echo "$TAG sam2 FAILED"; tail -5 "$CLIP/sam2.log"; continue; }

    say "[$INDEX/$TOTAL] $TAG: foundation pose"
    python -m v2d.foundation_pose.docker.run_video_to_poses \
      --video_path "$VIDEO" --depth_folder "$CLIP/depth" --masks_folder "$CLIP/masks" \
      --camera_intrinsics_path "$INTR" --mesh_path "$MESH_FILE" --poses_dir "$POSES" \
      --weights_dir "$WEIGHTS/foundation_pose" --backend nvlabs_pytorch --reference_frame 0 \
      > "$CLIP/fpose.log" 2>&1 || { echo "$TAG fpose FAILED"; tail -5 "$CLIP/fpose.log"; continue; }

    # The camera-frame track, plus the intrinsics and gravity, travel home; the world alignment is
    # free on the Jetson and is the one step that reads Tier 1.
    tar -C "$CLIP" -czf "$OUT/${TAG}_camera_frame.tgz" poses calib intr 2>/dev/null \
      && echo "$TAG exported $(stat -c%s "$OUT/${TAG}_camera_frame.tgz") bytes" \
      && mark "clip_$TAG"
  done < "$PLAN"
fi

say "summary"
ls -1 "$OUT" 2>/dev/null | wc -l | xargs echo "artefacts in $OUT:"
ls -1 "$MARK" | sed 's/^/  done: /'
echo "elapsed $(elapsed_h) h of a ${DEADLINE_H} h deadline"
