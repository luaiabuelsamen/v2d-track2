#!/usr/bin/env bash
# Tier 3 object reconstruction on ONE A6000, end to end:
#   create box -> build MegaPose -> refiner tracking for all 30 clips -> pull npz -> DELETE box
#   -> kit score per clip against Tier 1 truth (on the Jetson, free) -> md5 manifest.
#
# DRY RUN BY DEFAULT: without --go it runs only the free preflight, prints every step with its
# exact command, and prints the estimate. It creates nothing and spends nothing.
#
# The frame-0 camera solve is NOT done on the box. It is free on the Jetson
# (brev/t3_camera.py, ~35 s per clip) and must exist before launch in $CAM_DIR.
#
# Per clip on the box (MegaPose megapose-1.0-RGB refiner only, no coarse model):
#   init  = cam_T_world @ Tier1 world_T_obj(frame 0)   (FAQ PR 207: frame-0 alignment)
#   lock  = LOCKON_ITERS refiner iterations on frame 0; then the camera is corrected so the
#           locked pose equals the Tier 1 frame-0 pose (alignment only)
#   track = N_ITERS refiner iterations per frame, from the previous pose
#   write episode_NNNNNN.npz: pose_xyzw [T,1,7], coordinate_frame=released_track2_world
# The box stops tracking new clips at its own deadline (BOX_DEADLINE_H), so results are pulled
# before the money cap; clips it did not reach are reported as missing, never padded.
#
# Spend control, in order:
#   1. budget_guard.sh refuses before create if MAX_USD would breach the shared cap
#   2. a detached backstop (backstop_watchdog.sh) deletes the box at WATCHDOG_H no matter what
#   3. the EXIT trap deletes it on every exit path, then records billed minutes in the ledger
#
# Usage:
#   bash brev/t3_recon.sh            # dry run: preflight + plan + estimate
#   bash brev/t3_recon.sh --go       # spends money; only with Luai's approval for MAX_USD
set -uo pipefail

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_HAND=/tmp/claude-1000/-home-jetson3-projects-contact-insertion/5be04aec-f270-4096-8ae3-8e871a25b215/scratchpad
export V2D_SCRATCH=/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad
DATA=${DATA:-$V2D_SCRATCH/data}
KIT=${KIT:-$V2D_SCRATCH/kit_extract/v2d_submission_kit}
CAM_DIR=${CAM_DIR:-$SCRATCH_HAND/t3_cameras}
OUT=${OUT:-$SCRATCH_HAND/t3_recon}
NAME=${NAME:-v2d-t3recon-01}
INSTANCE_TYPE=${INSTANCE_TYPE:-massedcompute_A6000_plus}
RATE_USD_PER_HOUR=${RATE_USD_PER_HOUR:-0.68}
MAX_USD=${MAX_USD:-1.50}
N_ITERS=${N_ITERS:-3}
LOCKON_ITERS=${LOCKON_ITERS:-30}
MEGAPOSE_COMMIT=f3b8e12
HF=https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main
VIDEOS=track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera
WEIGHT_MIRRORS=(
  "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models"
  "https://gepettoweb.laas.fr/data/happypose/megapose/megapose-models"
)
WEIGHT_RUN=refiner-rgb-653307694
LOG=$OUT/launch.log

# Hours, all derived from the cap so the three limits cannot disagree.
WATCHDOG_H=$(awk -v u="$MAX_USD" -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", u/r}')
BOX_DEADLINE_H=$(awk -v w="$WATCHDOG_H" 'BEGIN{printf "%.2f", w-0.40}')  # pull + delete margin

say() { printf '\n== %s\n' "$*"; }
step() {  # step <description> -- <command...>: print always, execute only with --go
  local desc=$1; shift; [[ $1 == -- ]] && shift
  printf '   %-34s %s\n' "$desc" "$*"
  [[ $GO == 1 ]] && "$@"
  return 0
}

estimate() {
  # Minutes per stage. Tracking: 22,990 frames x N_ITERS x 66.5 ms per refiner iteration
  # (GenFlow's measured MegaPose refiner on an RTX 3090; A6000 taken as equal; not measured).
  awk -v n="$N_ITERS" -v r="$RATE_USD_PER_HOUR" -v l="$LOCKON_ITERS" 'BEGIN{
    create=8; setup=12; upload=6; videos=2; weights=2; lock=30*l*0.0665/60; pull=3;
    track=22990*n*0.0665/60;
    total=create+setup+upload+videos+weights+lock+track+pull;
    printf "   create+SSH ready      %5.1f min\n", create;
    printf "   build MegaPose env    %5.1f min\n", setup;
    printf "   upload inputs         %5.1f min  (meshes 345 MB, Tier 1 frame-0 poses, cameras)\n", upload;
    printf "   videos from HF        %5.1f min  (686 MB)\n", videos;
    printf "   weights               %5.1f min\n", weights;
    printf "   lock-on, 30 clips     %5.1f min\n", lock;
    printf "   tracking              %5.1f min  (22,990 frames x %d iters x 66.5 ms)\n", track, n;
    printf "   pull + delete         %5.1f min\n", pull;
    printf "   TOTAL                 %5.2f h  ->  $%.2f at $%.2f/h\n", total/60, total/60*r, r }'
}

# --------------------------------------------------------------------------- preflight (free)
mkdir -p "$OUT"
say "Preflight (free)"
PRE_OK=1
n_cam=$(ls "$CAM_DIR"/camera_e*.json 2>/dev/null | wc -l)
echo "   cameras         $n_cam/30 in $CAM_DIR"; [[ $n_cam == 30 ]] || PRE_OK=0
for d in "$DATA/track_2/tier_1_multiview_caption/meta" "$DATA/track_2/tier_1_multiview_caption/mesh" "$KIT/metric_code/track_2"; do
  [[ -d $d ]] && echo "   present         $d" || { echo "   MISSING         $d"; PRE_OK=0; }
done
WEIGHTS_FROM=""
for base in "${WEIGHT_MIRRORS[@]}"; do
  code=$(curl -s -o /dev/null -m 30 -I -w '%{http_code}' "$base/$WEIGHT_RUN/checkpoint.pth.tar" || true)
  echo "   weights         $code  $base/$WEIGHT_RUN/checkpoint.pth.tar"
  [[ $code == 200 || $code == 30? ]] && { WEIGHTS_FROM=$base; break; }
done
[[ -n $WEIGHTS_FROM ]] || { echo "   weights         UNREACHABLE on every mirror"; PRE_OK=0; }
~/.local/bin/brev ls >/dev/null 2>&1 && echo "   brev            logged in" || { echo "   brev            NOT logged in"; PRE_OK=0; }
free_gb=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9); echo "   disk            ${free_gb} GB free"

say "Estimate (one $INSTANCE_TYPE at \$$RATE_USD_PER_HOUR/h, N_ITERS=$N_ITERS)"
estimate
echo "   hard caps       box deadline ${BOX_DEADLINE_H} h, backstop delete ${WATCHDOG_H} h = \$$MAX_USD"

# --------------------------------------------------------------------------- the run
say "Steps ($([[ $GO == 1 ]] && echo EXECUTING || echo 'dry run: printed, not executed'))"
if [[ $GO == 1 && $PRE_OK != 1 ]]; then
  echo "!! preflight failed; refusing to create a box" >&2; exit 2
fi

source "$REPO_ROOT/brev/budget_guard.sh"
step "1  budget guard"                 -- budget_ok_for_hours "$WATCHDOG_H" "$NAME"
arm_backstop() {  # detached: outlives this script, deletes the box at WATCHDOG_H regardless
  setsid nohup bash "$REPO_ROOT/brev/backstop_watchdog.sh" "$NAME" \
    "$(awk -v h="$WATCHDOG_H" 'BEGIN{printf "%d", h*3600}')" "$LOG" </dev/null >/dev/null 2>&1 &
}
step "2  arm backstop (detached)"      -- arm_backstop
CREATED=0; T0=$SECONDS
teardown() {
  if [[ $CREATED == 1 ]]; then
    for _ in 1 2 3 4 5; do ~/.local/bin/brev delete "$NAME" </dev/null >>"$LOG" 2>&1 && break; sleep 60; done
    ab4_ledger_add "$NAME" "$(( (SECONDS - T0) / 60 + 1 ))"
    ~/.local/bin/brev ls 2>/dev/null | grep -q "^[[:space:]]*$NAME[[:space:]]" \
      && echo "!! $NAME STILL LISTED -- run: brev delete $NAME" >&2
  fi
}
[[ $GO == 1 ]] && trap teardown EXIT INT TERM
step "3  create box"                   -- ~/.local/bin/brev create "$NAME" --type "$INSTANCE_TYPE"
[[ $GO == 1 ]] && CREATED=1
wait_ssh() {
  for _ in $(seq 1 80); do
    ~/.local/bin/brev refresh >/dev/null 2>&1
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes "$NAME" true \
      >/dev/null 2>&1 && return 0
    sleep 15
  done
  echo "!! $NAME never accepted SSH" >&2; return 1
}
step "4  wait for SSH (<=20 min)"      -- wait_ssh

stage_inputs() {  # small bundle: meshes, meta, Tier 1 frame-0 poses only, cameras
  local b=$OUT/inputs; rm -rf "$b"; mkdir -p "$b/cameras"
  cp "$CAM_DIR"/camera_e*.json "$b/cameras/"
  cp -r "$DATA/track_2/tier_1_multiview_caption/mesh" "$DATA/track_2/tier_1_multiview_caption/meta" "$b/"
  "$REPO_ROOT/.venv311/bin/python" -I - "$DATA" "$b/frame0_poses.json" <<'PY'
import json, sys, pyarrow.parquet as pq
root, out = sys.argv[1], sys.argv[2]
d = {e: pq.read_table(f"{root}/track_2/tier_1_multiview_caption/data/chunk-000/episode_{e:06d}.parquet",
                      columns=["observation.object.pose"]).slice(0, 1)["observation.object.pose"][0].as_py()
     for e in range(30)}
json.dump(d, open(out, "w"))
PY
  tar -C "$OUT" -czf "$OUT/inputs.tgz" inputs
}
step "5  stage inputs (local)"         -- stage_inputs
step "6  upload inputs + runner"       -- scp -o ServerAliveInterval=30 "$OUT/inputs.tgz" \
       "$REPO_ROOT/brev/t3_recon_box.sh" "$NAME":~/
step "7  run on box (deadline ${BOX_DEADLINE_H} h)" -- ssh -o ServerAliveInterval=30 "$NAME" \
       "WEIGHTS_FROM='$WEIGHTS_FROM' N_ITERS=$N_ITERS LOCKON_ITERS=$LOCKON_ITERS DEADLINE_H=$BOX_DEADLINE_H MEGAPOSE_COMMIT=$MEGAPOSE_COMMIT bash ~/t3_recon_box.sh"
step "8  pull results"                 -- scp -o ServerAliveInterval=30 "$NAME:~/t3_out/*" "$OUT/"
step "9  delete box"                   -- teardown
[[ $GO == 1 ]] && { CREATED=0; trap - EXIT INT TERM; }

score_all() {  # free, on the Jetson: kit metric code via v2d score, per clip
  cd "$REPO_ROOT" || return 1
  local eps=() f b
  for f in "$OUT"/episode_*.npz; do b=${f##*_}; eps+=("$((10#${b%.npz}))"); done
  env -u PYTHONPATH PYTHONPATH=src .venv311/bin/python -m v2d.cli score --kit-root "$KIT" \
    --dataset-root "$DATA" --pred "npz:$OUT" --tier tier2 --per-episode --episodes "${eps[@]}" \
    -o "$OUT/kit_score.json"
}
step "10 kit score per clip (Jetson)"  -- score_all
step "11 md5 manifest"                 -- bash -c "cd '$OUT' && md5sum episode_*.npz > MD5SUMS"

say "Exact command"
echo "   bash $REPO_ROOT/brev/t3_recon.sh --go"
[[ $PRE_OK == 1 ]] || echo "   (would refuse right now: preflight failed above)"
exit 0
