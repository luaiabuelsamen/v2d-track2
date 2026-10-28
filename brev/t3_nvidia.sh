#!/usr/bin/env bash
# Tier 3 reconstruction on one box, with NVIDIA's own modules. See docs/T3_RECON.md.
#
# Tier 3 ships no trajectory, so the reconstruction IS the deliverable input. The chain and the
# reason it replaced the MegaPose design (whose only weight host is down) are in the design doc;
# this is the driver: stage inputs, create one box, build the five images, download the weights,
# reconstruct every clip, pull the camera-frame tracks, delete the box.
#
# It KEEPS THE BOX on a build or weights failure, inside the backstop watchdog. Five docker builds
# on rented time is exactly the case where create-run-delete burns the build again on every
# attempt, which cost us an afternoon on arm CR.
#
# DRY RUN BY DEFAULT: without --go it stages locally, prints every step and the estimate, and
# creates nothing.
#
#   bash brev/t3_nvidia.sh            # dry run
#   bash brev/t3_nvidia.sh --go       # spends money
set -uo pipefail

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export V2D_SCRATCH="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
S=$V2D_SCRATCH
DATA=${DATA:-$S/data}
VIDEOS=${VIDEOS:-$S/t3/videos_dl/track_2/tier_2_synthetic_noise/videos/chunk-000/observation.images.exo_camera}
OUT=${OUT:-$S/t3/out}
NAME=${NAME:-v2d-t3recon-01}
INSTANCE_TYPE=${INSTANCE_TYPE:-massedcompute_A6000_plus}
RATE=${RATE:-0.68}
BOX_DEADLINE_H=${BOX_DEADLINE_H:-4.0}
WATCHDOG_H=${WATCHDOG_H:-5.5}
MAX_USD=$(awk -v h="$WATCHDOG_H" -v r="$RATE" 'BEGIN{printf "%.2f", h*r}')
LOG=$OUT/driver.log
PLAN=$OUT/t3_plan.txt
mkdir -p "$OUT"

say() { printf '\n== %s\n' "$*"; }
step() {
  local desc=$1; shift; [[ ${1:-} == -- ]] && shift
  printf '   %-34s %s\n' "$desc" "$*"
  [[ $GO == 1 ]] || return 0
  "$@" >>"$LOG" 2>&1 || { echo "!! FAILED: $desc (see $LOG)" >&2; return 1; }
}

# --------------------------------------------------------------------------- plan (free)
# "<ep> <sequence_id> <object_name>", from the released per-episode metadata, which names the
# object and its mesh outright: {"object": "big_black_box", "mesh": "mesh/big_black_box/..."}.
# Deriving it from the sequence id instead does not work -- e05's sequence is `g1_box_rotate_01`
# and its mesh is `big_black_box` -- and guessing the object of the clip you are reconstructing is
# the wrong kind of guess to leave in a pipeline.
#
# Only `object` and `mesh` are read. The same rows carry a `task` caption, which is exactly what
# the FAQ bars from Tier 3, so it is not read; and the copy read here is TIER 2's, whose metadata
# the FAQ clause does not speak about at all.
say "Plan (free): the 30 clips and their objects"
MESH_DIR="$DATA/track_2/tier_1_multiview_caption/mesh"
META="$DATA/track_2/tier_2_synthetic_noise/meta/episodes_metadata.jsonl"
META="$META" MESH_DIR="$MESH_DIR" "$REPO_ROOT/.venv311/bin/python" -I - > "$PLAN" <<'PYEOF' || exit 2
import json
import os
import pathlib

mesh_root = pathlib.Path(os.environ["MESH_DIR"])
problems = []
rows = 0
for line in open(os.environ["META"]):
    if not line.strip():
        continue
    row = json.loads(line)
    episode, sequence = int(row["episode_index"]), row["sequence_id"]
    name = row.get("object")
    if not name:
        problems.append(f"e{episode:02d} has no object in its metadata")
        continue
    if not (mesh_root / name).is_dir():
        problems.append(f"e{episode:02d} names object {name!r} but {mesh_root / name} is not there")
        continue
    print(f"{episode} {sequence} {name}")
    rows += 1
if problems:
    raise SystemExit("; ".join(problems))
if rows != 30:
    raise SystemExit(f"expected 30 episodes in the metadata, read {rows}")
PYEOF
printf '   %-15s %s\n' "clips" "$(grep -c . "$PLAN") with an object resolved"
printf '   %-15s %s\n' "objects" "$(awk '{print $3}' "$PLAN" | sort -u | wc -l) distinct"

# --------------------------------------------------------------------------- preflight (free)
say "Preflight (free)"
PRE_OK=1
VID_COUNT=$(ls "$VIDEOS"/episode_*.mp4 2>/dev/null | wc -l)
printf '   %-15s %s\n' "videos" "$VID_COUNT/30 in $VIDEOS"
[[ $VID_COUNT == 30 ]] || PRE_OK=0
printf '   %-15s %s\n' "meshes" "$(ls -d "$MESH_DIR"/*/ 2>/dev/null | wc -l) directories"
[[ -d $MESH_DIR ]] || PRE_OK=0
~/.local/bin/brev ls >/dev/null 2>&1 && printf '   %-15s %s\n' "brev" "logged in" \
  || { echo "   brev            NOT logged in"; PRE_OK=0; }
printf '   %-15s %s\n' "disk" "$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9) GB free"

say "Estimate (one $INSTANCE_TYPE at \$$RATE/h)"
cat <<EST
   create + SSH ready        8 min
   5 docker images          75 min  (geocalib, moge, grounding_dino, sam2, foundation_pose)
   weights                  12 min  (foundation_pose on the nvlabs_pytorch backend: no EULA flag)
   upload videos + meshes    8 min  (655 MB of video, 345 MB of mesh)
   30 clips x 5 stages      90 min  (23k frames: depth is the slow stage, tracking ~30 ms/frame)
   pull + delete             5 min
   TOTAL                   ~3.3 h  ->  \$$(awk -v r="$RATE" 'BEGIN{printf "%.2f", 3.3*r}')
   hard caps       box deadline ${BOX_DEADLINE_H} h, backstop delete ${WATCHDOG_H} h = \$$MAX_USD
   on a build failure the box is KEPT for fixing in place, still under the backstop
EST

# --------------------------------------------------------------------------- the run
say "Steps ($([[ $GO == 1 ]] && echo EXECUTING || echo 'dry run: printed, not executed'))"
if [[ $GO == 1 && $PRE_OK != 1 ]]; then
  echo "!! preflight failed; refusing to create a box" >&2; exit 2
fi

source "$REPO_ROOT/brev/budget_guard.sh"
step "1  budget guard"                 -- budget_ok_for_hours "$WATCHDOG_H" "$NAME"
arm_backstop() {
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

upload() {
  ssh -o BatchMode=yes "$NAME" 'mkdir -p ~/t3_videos ~/t3_meshes' || return 1
  rsync -a -e "ssh -o ServerAliveInterval=30" "$VIDEOS"/ "$NAME":~/t3_videos/ || return 1
  rsync -a -e "ssh -o ServerAliveInterval=30" "$MESH_DIR"/ "$NAME":~/t3_meshes/ || return 1
  scp -o ServerAliveInterval=30 "$PLAN" "$NAME":~/t3_plan.txt || return 1
  scp -o ServerAliveInterval=30 "$REPO_ROOT/brev/t3_nvidia_box.sh" "$NAME":~/ || return 1
}
step "5  upload videos, meshes, plan"  -- upload

# Build and weights FIRST, as their own step: if they fail the box is kept and nothing has been
# spent attempting 30 clips against a broken chain.
build_and_weights() {
  ssh -o ServerAliveInterval=30 "$NAME" "STAGE=build DEADLINE_H=$BOX_DEADLINE_H bash ~/t3_nvidia_box.sh" \
    || return $?
  ssh -o ServerAliveInterval=30 "$NAME" "STAGE=weights DEADLINE_H=$BOX_DEADLINE_H bash ~/t3_nvidia_box.sh"
}
if ! step "6  build 5 images + weights" -- build_and_weights; then
  if [[ $GO == 1 ]]; then
    echo "!! the chain did not build. The box $NAME is KEPT (backstop deletes it at ${WATCHDOG_H} h)." >&2
    echo "   Look with:  ssh $NAME 'tail -40 ~/t3/build_*.log ~/t3/weights_*.log'" >&2
    echo "   Then:       ssh $NAME 'STAGE=all bash ~/t3_nvidia_box.sh'   (stages already done are skipped)" >&2
    exit 3
  fi
fi

step "7  reconstruct 30 clips"         -- ssh -o ServerAliveInterval=30 "$NAME" \
       "STAGE=clips DEADLINE_H=$BOX_DEADLINE_H bash ~/t3_nvidia_box.sh"
step "8  pull camera-frame tracks"     -- rsync -a -e "ssh -o ServerAliveInterval=30" \
       "$NAME":~/t3_out/ "$OUT/tracks/"
step "9  delete box"                   -- teardown
[[ $GO == 1 ]] && CREATED=0

if [[ $GO == 1 ]]; then
  say "Pulled $(ls -1 "$OUT/tracks" 2>/dev/null | wc -l) artefacts to $OUT/tracks"
  echo "Next, free on the Jetson: frame-0 alignment into the released world (FAQ PR 207), then"
  echo "score each reconstruction against Tier 1 truth before any policy trains on it."
else
  say "Exact command"
  echo "   bash $REPO_ROOT/brev/t3_nvidia.sh --go"
  [[ $PRE_OK == 1 ]] || echo "   (would refuse right now: preflight failed above)"
fi
