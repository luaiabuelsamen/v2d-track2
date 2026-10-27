#!/usr/bin/env bash
# Tier 3 object reconstruction on ONE A6000: MegaPose RGB-only refiner tracking, all 30 clips.
#
# DRY RUN BY DEFAULT. Without --go this prints the plan and the cost estimate and touches
# nothing. It never creates, starts or stops a box: run it ON an instance someone else
# launched, after Luai has approved the spend (no paid boxes without approval).
#
# Inputs, prepared on the Jetson before the box exists (free; ~35 s per clip there):
#   $CAM_DIR/camera_eNN.json  {"K": 3x3 full-res, "world_T_cam": 4x4}, from the frame-0 camera
#       solve (ViTPose keypoints vs SOMA joints: Tier 1 on frame 0 as FAQ PR 207 allows, Tier 2
#       human elsewhere). Produced by ws/hand's t3 camera step.
#   $DATA  challenge dataset root holding track_2/tier_1_multiview_caption (meshes, meta, and
#       frame 0 of observation.object.pose, the only Tier 1 pose read here).
#
# Method per clip (mirrors the Jetson M3T smoke test in ws/hand docs/TIER3_FEASIBILITY.md):
#   1. init = cam_T_world @ Tier1 world_T_obj(frame 0)
#   2. lock-on: LOCKON_ITERS refiner iterations on frame 0, then correct the camera so the
#      locked frame-0 pose equals the Tier 1 frame-0 pose (alignment only, as the FAQ allows)
#   3. track: per frame, N_ITERS refiner iterations from the previous pose (no coarse model)
#   4. write $OUT/episode_NNNNNN.npz: pose_xyzw [T,1,7], coordinate_frame=released_track2_world
#
# Estimate (not measured on an A6000):
#   frames            22,990 (30 clips, 440-1,003 each)
#   refiner           66.5 ms per iteration on an RTX 3090 (GenFlow's MegaPose timing); A6000
#                     taken as equal. N_ITERS=5 -> ~0.33 s/frame -> ~2.1 h
#   lock-on           30 clips x 30 iterations -> < 2 min
#   setup             image pull/build + weights + 686 MB of video: ~0.5-0.75 h
#   total             ~2.75-3 h, budget 4 h with margin
#   cost at $0.68/h   ~$1.90-2.05 expected, $2.72 at the 4 h cap
#
# Known risk: the MegaPose weights host (www.paris.inria.fr) timed out on 2026-10-27 and the
# LAAS and BOP mirrors return 404 for refiner-rgb-653307694. The script checks the weights
# BEFORE anything long runs and stops if it cannot get them.
#
# The Jetson smoke test of the same structure with M3T's region tracker (no learned weights)
# scored ADD-AUC 0.044 on e24 at 12 ms/frame. A hold-the-frame-0-pose baseline scores 0.387.
# Monocular silhouettes do not pin depth at ~3.5 m. MegaPose's learned refiner is the remaining
# untested candidate; this script exists so that test costs ~$2, not so that it runs by default.
#
# Usage:  bash brev/t3_recon.sh                     # dry run: plan + estimate
#         bash brev/t3_recon.sh --go                # on an approved box only
set -euo pipefail

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

DATA=${DATA:-$HOME/v2d_data}
CAM_DIR=${CAM_DIR:-$HOME/t3_cameras}
OUT=${OUT:-$HOME/tier3_refs}
WORK=${WORK:-$HOME/t3_work}
EPISODES=${EPISODES:-$(seq -s ' ' 0 29)}
N_ITERS=${N_ITERS:-5}
LOCKON_ITERS=${LOCKON_ITERS:-30}
RATE_USD_H=${RATE_USD_H:-0.68}
MAX_HOURS=${MAX_HOURS:-4}
MEGAPOSE_COMMIT=f3b8e12
HF=https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main
VIDEOS=track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera
WEIGHT_MIRRORS=(
  "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models"
  "https://gepettoweb.laas.fr/data/happypose/megapose/megapose-models"
)

plan() {
  local n_ep
  n_ep=$(wc -w <<<"$EPISODES")
  cat <<EOF
t3_recon plan (dry run: nothing executed)
  episodes        $n_ep: $EPISODES
  data            $DATA  (meshes, meta, Tier 1 frame-0 pose only)
  cameras         $CAM_DIR/camera_eNN.json (from the Jetson)
  output          $OUT/episode_NNNNNN.npz + MD5SUMS (ms/frame printed per clip)
  model           MegaPose megapose-1.0-RGB refiner (refiner-rgb-653307694), commit $MEGAPOSE_COMMIT
  per frame       $N_ITERS refiner iterations, lock-on $LOCKON_ITERS on frame 0
  estimate        ~0.33 s/frame x 22,990 frames ~ 2.1 h + ~0.75 h setup ~ 2.9 h
  cost            ~\$$(python3 -c "print(f'{2.9*$RATE_USD_H:.2f}')") at \$$RATE_USD_H/h; cap $MAX_HOURS h = \$$(python3 -c "print(f'{$MAX_HOURS*$RATE_USD_H:.2f}')")
  steps           1 check GPU, disk, inputs   2 fetch weights (abort if unreachable)
                  3 build MegaPose env        4 fetch videos (HF, 686 MB)
                  5 track each clip           6 md5 manifest
EOF
}

run() { if [[ $GO == 1 ]]; then "$@"; else echo "  would run: $*"; fi; }

plan
if [[ $GO == 0 ]]; then
  echo
  echo "commands (dry run):"
fi

# 1. checks
run nvidia-smi --query-gpu=name,memory.total --format=csv
run test -d "$DATA/track_2/tier_1_multiview_caption/meta"
run test -d "$CAM_DIR"
run mkdir -p "$OUT" "$WORK"

# 2. weights first: the long steps are pointless without them
fetch_weights() {
  local dest=$WORK/megapose_data/megapose-models/refiner-rgb-653307694
  mkdir -p "$dest"
  for base in "${WEIGHT_MIRRORS[@]}"; do
    if curl -fsS -m 60 -o "$dest/checkpoint.pth.tar" "$base/refiner-rgb-653307694/checkpoint.pth.tar" &&
       curl -fsS -m 60 -o "$dest/config.yaml" "$base/refiner-rgb-653307694/config.yaml"; then
      echo "weights from $base"
      return 0
    fi
  done
  echo "MegaPose refiner weights unreachable on every mirror; stopping before any long step" >&2
  return 1
}
run fetch_weights

# 3. environment (x86-64 + CUDA; MegaPose's renderer is panda3d)
setup_env() {
  cd "$WORK"
  [[ -d megapose6d ]] || git clone -q https://github.com/megapose6d/megapose6d.git
  git -C megapose6d checkout -q "$MEGAPOSE_COMMIT"
  python3 -m venv venv && . venv/bin/activate
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -q panda3d panda3d-gltf trimesh omegaconf transforms3d roma pyarrow opencv-python-headless \
    simplejson pypng bokeh pandas scipy tqdm webdataset joblib
  pip install -q -e megapose6d
}
run setup_env

# 4. videos
fetch_videos() {
  mkdir -p "$WORK/videos"
  for e in $EPISODES; do
    f=episode_$(printf %06d "$e").mp4
    [[ -s "$WORK/videos/$f" ]] || curl -fsSL --retry 3 -o "$WORK/videos/$f" "$HF/$VIDEOS/$f"
  done
}
run fetch_videos

# 5. track (driver below; budget guard stops at MAX_HOURS)
track_all() {
  . "$WORK/venv/bin/activate"
  export MEGAPOSE_DATA_DIR=$WORK/megapose_data
  local start=$SECONDS
  for e in $EPISODES; do
    if (( SECONDS - start > MAX_HOURS * 3600 )); then echo "budget cap reached at e$e" >&2; break; fi
    python3 "$WORK/t3_megapose_track.py" --episode "$e" --data "$DATA" \
      --camera "$CAM_DIR/camera_e$(printf %02d "$e").json" \
      --video "$WORK/videos/episode_$(printf %06d "$e").mp4" --out "$OUT" \
      --n-iters "$N_ITERS" --lockon-iters "$LOCKON_ITERS"
  done
}
write_driver() {
  cat > "$WORK/t3_megapose_track.py" <<'PY'
"""MegaPose RGB refiner tracking for one clip (untested: weights were unreachable when written)."""
import argparse, json, time
from pathlib import Path
import cv2, numpy as np, pyarrow.parquet as pq, torch
from scipy.spatial.transform import Rotation as R
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.inference.types import ObservationTensor, PoseEstimatesType
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.tensor_collection import PandasTensorCollection
import pandas as pd

ap = argparse.ArgumentParser()
for a in ("--episode", "--n-iters", "--lockon-iters"):
    ap.add_argument(a, type=int, required=True)
for a in ("--data", "--camera", "--video", "--out"):
    ap.add_argument(a, type=Path, required=True)
args = ap.parse_args()
t1 = args.data / "track_2/tier_1_multiview_caption"
meta = next(json.loads(l) for l in open(t1 / "meta/episodes_metadata.jsonl")
            if l.strip() and json.loads(l)["episode_index"] == args.episode)
p = np.asarray(pq.read_table(t1 / f"data/chunk-000/episode_{args.episode:06d}.parquet",
               columns=["observation.object.pose"])["observation.object.pose"][0].as_py())
world_T_obj0 = np.eye(4); world_T_obj0[:3, :3] = R.from_quat([*p[4:7], p[3]]).as_matrix(); world_T_obj0[:3, 3] = p[:3]
cam = json.loads(args.camera.read_text())
K = np.asarray(cam["K"], np.float32); world_T_cam = np.asarray(cam["world_T_cam"])
label = meta["object"]
objects = RigidObjectDataset([RigidObject(label=label, mesh_path=t1 / meta["mesh"], mesh_units="m")])
model = load_named_model("megapose-1.0-RGB", objects).cuda()

def refine(rgb, TCO, n):
    obs = ObservationTensor.from_numpy(rgb, None, K).cuda()
    data = PandasTensorCollection(infos=pd.DataFrame({"label": [label], "batch_im_id": [0], "instance_id": [0]}),
                                  poses=torch.as_tensor(TCO[None], dtype=torch.float32).cuda())
    out, _ = model.forward_refiner(obs, data, n_iterations=n, keep_all_outputs=False)
    return out[f"iteration={n}"].poses[0].cpu().numpy().astype(np.float64)

cap = cv2.VideoCapture(str(args.video)); frames = []
ok, bgr = cap.read()
rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
TCO = refine(rgb0, np.linalg.inv(world_T_cam) @ world_T_obj0, args.lockon_iters)
world_T_cam = world_T_obj0 @ np.linalg.inv(TCO)  # frame-0 alignment (FAQ PR 207)
poses, times = [TCO], []
while True:
    ok, bgr = cap.read()
    if not ok: break
    t = time.time(); TCO = refine(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), TCO, args.n_iters)
    torch.cuda.synchronize(); times.append(time.time() - t); poses.append(TCO)
W = np.einsum("ij,tjk->tik", world_T_cam, np.asarray(poses))
x = np.concatenate([W[:, :3, 3], R.from_matrix(W[:, :3, :3]).as_quat()], 1)[:, None]
args.out.mkdir(parents=True, exist_ok=True)
np.savez(args.out / f"episode_{args.episode:06d}.npz", episode_index=np.int64(args.episode),
         coordinate_frame=np.array("released_track2_world"), frame_index=np.arange(len(x)), pose_xyzw=x)
print(json.dumps({"episode": args.episode, "frames": len(x), "ms_per_frame": 1000 * float(np.mean(times))}))
PY
}
run write_driver
run track_all

# 6. manifest
run bash -c "cd '$OUT' && md5sum episode_*.npz > MD5SUMS"
[[ $GO == 0 ]] && echo && echo "dry run complete; pass --go on an approved box to execute"
exit 0
