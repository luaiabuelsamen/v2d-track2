#!/usr/bin/env bash
# Runs ON the A6000 box, launched by brev/t3_recon.sh. Never run it locally.
# Env from the launcher: WEIGHTS_FROM, N_ITERS, LOCKON_ITERS, DEADLINE_H, MEGAPOSE_COMMIT.
# Output: ~/t3_out/episode_NNNNNN.npz (+ timing_eNN.json), and box_status.json.
# Untested end to end: the MegaPose weights were unreachable when this was written.
set -uo pipefail
START=$SECONDS
WORK=~/t3_work
OUTB=~/t3_out
mkdir -p "$WORK" "$OUTB"
status() { printf '{"stage": "%s", "elapsed_s": %d}\n' "$1" $((SECONDS - START)) > "$OUTB/box_status.json"; echo "[$((SECONDS - START))s] $1"; }

status unpack
tar -C "$WORK" -xzf ~/inputs.tgz

status weights
D=$WORK/megapose_data/megapose-models/refiner-rgb-653307694
mkdir -p "$D"
curl -fsS -m 300 -o "$D/checkpoint.pth.tar" "$WEIGHTS_FROM/refiner-rgb-653307694/checkpoint.pth.tar" &&
curl -fsS -m 60 -o "$D/config.yaml" "$WEIGHTS_FROM/refiner-rgb-653307694/config.yaml" ||
  { status "weights download failed"; exit 3; }

status env
cd "$WORK"
git clone -q https://github.com/megapose6d/megapose6d.git && git -C megapose6d checkout -q "$MEGAPOSE_COMMIT"
python3 -m venv venv && . venv/bin/activate
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q panda3d panda3d-gltf trimesh omegaconf transforms3d roma pyarrow opencv-python-headless \
  simplejson pypng bokeh pandas scipy tqdm webdataset joblib
pip install -q -e megapose6d || { status "megapose install failed"; exit 4; }

status videos
HF=https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main
V=track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera
mkdir -p videos
for e in $(seq 0 29); do
  f=episode_$(printf %06d "$e").mp4
  curl -fsSL --retry 3 -o "videos/$f" "$HF/$V/$f" &
  (( e % 6 == 5 )) && wait
done
wait

cat > track.py <<'PY'
"""MegaPose RGB refiner tracking for one clip, from the frame-0 camera solve and Tier 1 frame-0 pose."""
import argparse, json, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from scipy.spatial.transform import Rotation as R
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.inference.types import ObservationTensor
from megapose.utils.load_model import load_named_model
from megapose.utils.tensor_collection import PandasTensorCollection

ap = argparse.ArgumentParser()
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--n-iters", type=int, required=True)
ap.add_argument("--lockon-iters", type=int, required=True)
ap.add_argument("--inputs", type=Path, required=True)
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
meta = next(json.loads(l) for l in open(a.inputs / "meta/episodes_metadata.jsonl")
            if l.strip() and json.loads(l)["episode_index"] == a.episode)
p = np.asarray(json.load(open(a.inputs / "frame0_poses.json"))[str(a.episode)], float)  # x y z qw qx qy qz
world_T_obj0 = np.eye(4); world_T_obj0[:3, :3] = R.from_quat(p[[4, 5, 6, 3]]).as_matrix(); world_T_obj0[:3, 3] = p[:3]
cam = json.load(open(a.inputs / f"cameras/camera_e{a.episode:02d}.json"))
K = np.asarray(cam["K"], np.float32); world_T_cam = np.asarray(cam["world_T_cam"], float)
label = meta["object"]
objects = RigidObjectDataset([RigidObject(label=label, mesh_path=a.inputs / meta["mesh"], mesh_units="m")])
model = load_named_model("megapose-1.0-RGB", objects).cuda()

def refine(rgb, TCO, n):
    obs = ObservationTensor.from_numpy(rgb, None, K).cuda()
    data = PandasTensorCollection(
        infos=pd.DataFrame({"label": [label], "batch_im_id": [0], "instance_id": [0]}),
        poses=torch.as_tensor(TCO[None], dtype=torch.float32).cuda())
    out, _ = model.forward_refiner(obs, data, n_iterations=n, keep_all_outputs=False)
    return out[f"iteration={n}"].poses[0].cpu().numpy().astype(np.float64)

cap = cv2.VideoCapture(str(a.video))
ok, bgr = cap.read()
TCO = refine(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.linalg.inv(world_T_cam) @ world_T_obj0, a.lockon_iters)
world_T_cam = world_T_obj0 @ np.linalg.inv(TCO)  # frame-0 alignment only
poses, times = [TCO], []
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    t = time.time()
    TCO = refine(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), TCO, a.n_iters)
    torch.cuda.synchronize(); times.append(time.time() - t); poses.append(TCO)
W = np.einsum("ij,tjk->tik", world_T_cam, np.asarray(poses))
x = np.concatenate([W[:, :3, 3], R.from_matrix(W[:, :3, :3]).as_quat()], 1)[:, None]
np.savez(a.out / f"episode_{a.episode:06d}.npz", episode_index=np.int64(a.episode),
         coordinate_frame=np.array("released_track2_world"), frame_index=np.arange(len(x)), pose_xyzw=x)
(a.out / f"timing_e{a.episode:02d}.json").write_text(json.dumps(
    {"episode": a.episode, "frames": len(x), "ms_per_frame": 1000 * float(np.mean(times))}))
print(a.episode, len(x), round(1000 * float(np.mean(times)), 1), "ms/frame", flush=True)
PY

export MEGAPOSE_DATA_DIR=$WORK/megapose_data
DEADLINE_S=$(awk -v h="$DEADLINE_H" 'BEGIN{printf "%d", h*3600}')
for e in $(seq 0 29); do
  if (( SECONDS - START > DEADLINE_S )); then status "deadline reached before e$e"; break; fi
  status "tracking e$e"
  python3 track.py --episode "$e" --n-iters "$N_ITERS" --lockon-iters "$LOCKON_ITERS" \
    --inputs "$WORK/inputs" --video "videos/episode_$(printf %06d "$e").mp4" --out "$OUTB" \
    >> "$OUTB/track.log" 2>&1 || echo "e$e failed" >> "$OUTB/track.log"
done
status done
