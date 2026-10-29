"""Frame-0 camera solve for the Tier 3 reconstruction (free; runs on the Jetson GPU).

Static exo camera, no released intrinsics. RT-DETR person box -> ViTPose COCO keypoints on
`--frames` frames spread over the clip, matched to SOMA joints: the Tier 1 human on frame 0 (FAQ
PR 207: the first Tier 1 frame may be used for alignment) and the Tier 2 human on the rest. PnP
RANSAC over a focal sweep (principal point at the image centre), then a Huber least-squares
refinement of focal and pose.

Writes camera_eNN.json: {"K": 3x3 at full resolution, "world_T_cam": 4x4 (released OpenCV
world), "reproj_median_px", "reproj_p90_px", "points"}. Tested on e24 (ws/hand smoke test):
reprojection median 3.9 px at half resolution.

    python t3_camera.py --videos DIR --human-cache DIR --out DIR [--episodes 0 1 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])  # its own inverse
COCO_TO_SOMA = {
    1: "LeftEye", 2: "RightEye", 5: "LeftArm", 6: "RightArm", 7: "LeftForeArm",
    8: "RightForeArm", 9: "LeftHand", 10: "RightHand", 11: "LeftLeg", 12: "RightLeg",
    13: "LeftShin", 14: "RightShin", 15: "LeftFoot", 16: "RightFoot",
}
SCALE = 0.5


class Keypoints:
    def __init__(self, device: str) -> None:
        from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

        self.device = device
        self.det_proc = AutoProcessor.from_pretrained("PekingU/rtdetr_r18vd_coco_o365")
        self.det = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd_coco_o365").to(device)
        self.pose_proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.pose = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device)

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        pil = Image.fromarray(image)
        r = self.det_proc(images=pil, return_tensors="pt").to(self.device)
        res = self.det_proc.post_process_object_detection(
            self.det(**r), target_sizes=torch.tensor([(pil.height, pil.width)]), threshold=0.4
        )[0]
        boxes = res["boxes"][res["labels"] == 0].cpu().numpy()
        if len(boxes) == 0:
            return None
        box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
        box = np.array([[box[0], box[1], box[2] - box[0], box[3] - box[1]]])
        p = self.pose_proc(pil, boxes=[box], return_tensors="pt").to(self.device)
        k = self.pose_proc.post_process_pose_estimation(self.pose(**p), boxes=[box])[0][0]
        return np.concatenate([k["keypoints"].cpu().numpy(), k["scores"].cpu().numpy()[:, None]], 1)


def read_frames(video: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    cap, out, f = cv2.VideoCapture(str(video)), {}, 0
    last = max(wanted)
    while f <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        if f in wanted:
            bgr = cv2.resize(bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            out[f] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        f += 1
    return out


def solve(kps: dict, joints: np.ndarray, names: list[str], size: tuple[int, int]) -> dict:
    w, h = size
    pts2, pts3 = [], []
    for f, k in kps.items():
        if k is None:
            continue
        for c, s in COCO_TO_SOMA.items():
            if k[c, 2] > 0.5:
                pts2.append(k[c, :2])
                pts3.append(joints[f][names.index(s)])
    pts2, pts3 = np.asarray(pts2, float), np.asarray(pts3, float)
    best = None
    for f in np.geomspace(0.4 * w, 4.0 * w, 60):
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(pts3, pts2, K, None, reprojectionError=12.0,
                                                 iterationsCount=200)
        if not ok or inl is None:
            continue
        err = np.linalg.norm(cv2.projectPoints(pts3, rvec, tvec, K, None)[0][:, 0] - pts2, axis=1)
        if best is None or np.median(err) < best[0]:
            best = (np.median(err), f, rvec.ravel(), tvec.ravel())

    def residual(x: np.ndarray) -> np.ndarray:
        K = np.array([[x[0], 0, w / 2], [0, x[0], h / 2], [0, 0, 1.0]])
        return (cv2.projectPoints(pts3, x[1:4], x[4:7], K, None)[0][:, 0] - pts2).ravel()

    sol = least_squares(residual, np.concatenate([[best[1]], best[2], best[3]]), loss="huber",
                        f_scale=8.0)
    err = np.linalg.norm(residual(sol.x).reshape(-1, 2), axis=1)
    cam_T_world = np.eye(4)
    cam_T_world[:3, :3] = Rotation.from_rotvec(sol.x[1:4]).as_matrix()
    cam_T_world[:3, 3] = sol.x[4:7]
    f_full = sol.x[0] / SCALE
    K_full = [[f_full, 0, w / SCALE / 2], [0, f_full, h / SCALE / 2], [0, 0, 1.0]]
    return {"K": K_full, "world_T_cam": np.linalg.inv(cam_T_world).tolist(),
            "reproj_median_px_half_res": float(np.median(err)),
            "reproj_p90_px_half_res": float(np.percentile(err, 90)), "points": int(len(pts2))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--human-cache", type=Path, required=True,
                    help="human_tier1_eNN.npz / human_tier2_eNN.npz (SOMA world joints)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = Keypoints("cuda" if torch.cuda.is_available() else "cpu")
    for e in args.episodes:
        target = args.out / f"camera_e{e:02d}.json"
        if target.exists():
            continue
        j1 = np.load(args.human_cache / f"human_tier1_e{e:02d}.npz")
        j2 = np.load(args.human_cache / f"human_tier2_e{e:02d}.npz")
        names = [str(n) for n in j1["names"]]
        joints = j2["joints"].astype(float) @ CV_TO_SOMA
        joints[0] = j1["joints"][0].astype(float) @ CV_TO_SOMA  # Tier 1 on frame 0 only
        wanted = sorted(set(np.linspace(0, len(joints) - 1, args.frames).astype(int)) | {0})
        images = read_frames(args.videos / f"episode_{e:06d}.mp4", wanted)
        h, w = next(iter(images.values())).shape[:2]
        result = solve({f: model(im) for f, im in images.items()}, joints, names, (w, h))
        result["episode"] = e
        target.write_text(json.dumps(result, indent=1) + "\n")
        print(f"e{e:02d} f={result['K'][0][0]:.1f}px median {result['reproj_median_px_half_res']:.2f}px", flush=True)


if __name__ == "__main__":
    main()
