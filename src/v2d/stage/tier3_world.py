"""Put a Tier 3 reconstruction into the released world frame, with one permitted anchor.

FoundationPose writes one `Transform3d` per frame in the **camera** frame, and no camera
calibration is released for Tier 3. The challenge FAQ (nvidia-isaac/video_to_data PR 207, merged)
settles how to bridge that:

    "since no camera calibration is released for Tier 3, teams may use the first frame of Tier 1
     data to compute alignment, provided the Tier 1 caption isn't otherwise used in Tier 3."

So the anchor is exactly one frame. The object's pose at frame 0 is known in both frames -- in the
camera frame from the reconstruction, in the world frame from Tier 1 -- and a single pose
correspondence determines a rigid transform outright:

    world_from_cam = world_from_obj[0] @ inv(cam_from_obj[0])

applied unchanged to every later frame. Nothing after frame 0 is read from Tier 1, no scale is
fitted (FoundationPose works in metres against the released mesh, so the scale is already right),
and no per-frame correction is applied -- which is what keeps this an alignment and not a quiet
regression onto the truth.

The quaternion convention is not guessed: `Transform3d.to_matrix` is imported from upstream's own
`v2d_common.datatypes`, so `rotation` is read as wxyz by the same code that wrote it.

    v2d tier3-world --poses DIR --dataset-root DATA --npz-out DIR [--episodes 0 1 ...]

Writes `episode_NNNNNN.npz` holding `pose_xyzw [T,7]`, which `v2d score --pred npz:DIR` reads, and
a record carrying the per-clip anchor residual and the frame count.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v2d.dataset import object_pose

UPSTREAM_COMMON = Path("/home/jetson3/projects/v2d-upstream/reconstruction/modules/v2d_common")


def _transform3d():
    """Upstream's own Transform3d, so the wxyz convention comes from the writer, not from me.

    Loaded straight off `datatypes.py` by path: importing the `v2d_common` package would run its
    `__init__`, which imports `v2d.common.broadcast` and collides with this repo's own `v2d`
    package. `datatypes.py` has no intra-package imports, so the file stands alone.
    """
    path = UPSTREAM_COMMON / "datatypes.py"
    if not path.is_file():
        raise SystemExit(f"FATAL: upstream datatypes.py is not at {path}")
    spec = importlib.util.spec_from_file_location("_v2d_upstream_datatypes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Transform3d


def camera_track(poses_dir: Path, transform3d) -> tuple[np.ndarray, list[int]]:
    """``(T,4,4)`` cam_from_obj in frame order, and the frame indices behind it."""
    files = sorted(poses_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"FATAL: no pose JSONs under {poses_dir}")
    frames, mats = [], []
    for path in files:
        with path.open() as handle:
            transform = transform3d.from_dict(json.load(handle))
        matrix = transform.to_matrix()
        if not np.isfinite(matrix).all():
            raise SystemExit(f"FATAL: {path} holds a non-finite pose")
        frames.append(int(path.stem))
        mats.append(matrix)
    return np.stack(mats), frames


def to_world(cam_poses: np.ndarray, world_from_obj0: np.ndarray) -> np.ndarray:
    """Apply the single frame-0 anchor to the whole camera-frame track."""
    world_from_cam = world_from_obj0 @ np.linalg.inv(cam_poses[0])
    return world_from_cam @ cam_poses


def as_pose_xyzw(world_poses: np.ndarray) -> np.ndarray:
    """``(T,7)`` xyz + xyzw, the layout the scorer reads."""
    quat = Rotation.from_matrix(world_poses[:, :3, :3]).as_quat()  # scipy returns xyzw
    return np.concatenate([world_poses[:, :3, 3], quat], axis=1)


def pose_matrix(pose_xyzw: np.ndarray) -> np.ndarray:
    """The inverse of `as_pose_xyzw` for a single ``(7,)`` pose."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose_xyzw[3:7]).as_matrix()
    matrix[:3, 3] = pose_xyzw[:3]
    return matrix


def align_episode(poses_root: Path, dataset_root: Path, episode: int, out_dir: Path) -> dict:
    transform3d = _transform3d()
    poses_dir = poses_root / f"e{episode:02d}" / "poses"
    cam_poses, frames = camera_track(poses_dir, transform3d)
    truth, _visible = object_pose(dataset_root, "tier1", episode)

    # The reconstruction runs on the released video, so a frame count that disagrees with the
    # released trajectory means frames were dropped and every later index is off by that much.
    if len(cam_poses) != len(truth):
        return {
            "episode": episode,
            "status": "FRAME COUNT MISMATCH",
            "reconstruction_frames": len(cam_poses),
            "released_frames": len(truth),
        }
    if frames != list(range(len(frames))):
        return {"episode": episode, "status": "NON-CONTIGUOUS FRAMES", "first_gap": frames[:5]}

    world = to_world(cam_poses, pose_matrix(truth[0]))
    pose_xyzw = as_pose_xyzw(world)

    # The anchor is exact by construction; measuring it is how a silent change to the convention
    # or the mesh scale would show up here rather than as a mysteriously poor score.
    anchor_cm = float(np.linalg.norm(pose_xyzw[0, :3] - truth[0, :3]) * 100.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    # `coordinate_frame` is what the retargeter and the scorer check before they will read this,
    # and the frame-0 anchor is exactly what earns the stamp: the poses are in the released world.
    np.savez_compressed(
        out_dir / f"episode_{episode:06d}.npz",
        pose_xyzw=pose_xyzw,
        coordinate_frame="released_track2_world",
    )
    return {
        "episode": episode,
        "status": "ok",
        "frames": len(pose_xyzw),
        "anchor_residual_cm": round(anchor_cm, 6),
        "mean_position_error_cm": round(
            float(np.linalg.norm(pose_xyzw[:, :3] - truth[:, :3], axis=1).mean() * 100.0), 3
        ),
        "final_position_error_cm": round(
            float(np.linalg.norm(pose_xyzw[-1, :3] - truth[-1, :3]) * 100.0), 3
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True, help="holds eNN/poses/*.json")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--npz-out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    episodes = args.episodes
    if not episodes:
        episodes = sorted(
            int(d.name[1:]) for d in args.poses.glob("e[0-9][0-9]") if (d / "poses").is_dir()
        )
    records = [align_episode(args.poses, args.dataset_root, ep, args.npz_out) for ep in episodes]

    ok = [r for r in records if r["status"] == "ok"]
    for record in records:
        if record["status"] != "ok":
            print(f"e{record['episode']:02d}  {record['status']}  {record}")
        else:
            print(
                f"e{record['episode']:02d}  {record['frames']} frames  "
                f"anchor {record['anchor_residual_cm']:.4f} cm  "
                f"mean {record['mean_position_error_cm']:7.2f} cm  "
                f"final {record['final_position_error_cm']:7.2f} cm"
            )
    print(f"\naligned {len(ok)} of {len(records)} episodes into {args.npz_out}")
    if ok:
        print(
            "mean position error across the aligned clips: "
            f"{np.mean([r['mean_position_error_cm'] for r in ok]):.2f} cm"
        )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "what": "Tier 3 reconstruction aligned into the released world frame",
                    "anchor": "frame 0 of Tier 1 only (FAQ PR 207); no later truth is read",
                    "episodes": records,
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if len(ok) == len(records) else 4


if __name__ == "__main__":
    sys.exit(main())
