"""Check a retargeted G1 motion parquet against the released object trajectory.

`soma_to_g1.py` levels the reconstructed ground plane to z=0 and applies a constant ground
lift, so the object trajectory it writes is the released one composed with **one rigid
transform** for the whole sequence. That is exactly the transform the submission packer undoes
later ("converts evaluator poses using one rigid transform between the evaluator's recorded
reference and that tier's released input"). So the retargeting is faithful if and only if a
single rigid transform explains the whole object trajectory.

Two checks, the second independent of any frame convention:

1. **Rigid fit.** Kabsch-fit one rotation+translation from the released object positions onto
   the retargeted ones and report the residual. A faithful retarget leaves sub-millimetre
   residual; a frame or ordering bug leaves centimetres or more.
2. **Relative rotations.** A rigid world transform cannot change the rotation *between* two
   frames of the trajectory. Comparing per-frame relative rotations therefore tests the
   orientation stream without knowing the transform at all.

Frames the release marks invisible are excluded from both checks. Such a frame is zero-filled,
and a zero quaternion reads back as the identity rotation at the world origin -- a point the
object was never at. Including even one drags the global Kabsch fit off by centimetres. The
converter carries the last visible pose across those frames, the evaluator does not score them
(episode 11's final frame is absent from the scored grid), so neither should this.

Usage:
  python scripts/validate_retarget.py --motion-parquet P --dataset-root DATA --episode 24 \
      [--tier tier1] [-o results/retarget_validation.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TIER_DIRS = {"tier1": "tier_1_multiview_caption", "tier2": "tier_2_synthetic_noise"}
# soma_to_g1.load_data: OpenCV world (X right, Y down, Z forward) -> SOMA world (X right, Y up)
CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (quat[..., i] for i in range(4))
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Best rigid transform taking `source` onto `target`; returns (R, t, residual RMS)."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    translation = target_mean - rotation @ source_mean
    residual = (rotation @ source.T).T + translation - target
    return rotation, translation, float(np.sqrt((residual**2).sum(axis=1).mean()))


def relative_rotation_angles(matrices: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) of the rotation between consecutive frames."""
    relative = np.einsum("tij,tkj->tik", matrices[1:], matrices[:-1])
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion-parquet", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--tier", default="tier1", choices=sorted(TIER_DIRS))
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    motion = pq.read_table(args.motion_parquet).to_pylist()[0]
    retarget_position = np.asarray(motion["object_body_position"], dtype=np.float64)[:, 0, :]
    retarget_wxyz = np.asarray(motion["object_body_wxyz"], dtype=np.float64)[:, 0, :]
    frames = len(retarget_position)

    released = pq.read_table(
        args.dataset_root / "track_2" / TIER_DIRS[args.tier] / "data" / "chunk-000"
        / f"episode_{args.episode:06d}.parquet",
        columns=["observation.object.pose", "observation.object.visible"],
    )
    pose = np.asarray(released["observation.object.pose"].to_pylist(), dtype=np.float64)[:frames]
    visible = np.asarray(released["observation.object.visible"].to_pylist(), dtype=bool)[:frames]
    pose, retarget_position_cmp, retarget_wxyz_cmp = pose[visible], retarget_position[visible], retarget_wxyz[visible]
    # released: xyz + wxyz in OpenCV world. Put it in the SOMA axes the retargeter works in.
    released_position = pose[:, :3] @ CV_TO_SOMA.T
    released_matrix = CV_TO_SOMA @ quat_wxyz_to_matrix(pose[:, 3:])

    rotation, translation, residual_m = kabsch(released_position, retarget_position_cmp)
    retarget_matrix = quat_wxyz_to_matrix(retarget_wxyz_cmp)
    angle_difference = np.abs(
        relative_rotation_angles(retarget_matrix) - relative_rotation_angles(released_matrix)
    )
    # The fitted rotation should also carry the released orientations onto the retargeted ones.
    carried = np.einsum("ij,tjk->tik", rotation, released_matrix)
    carried_error = np.einsum("tij,tkj->tik", retarget_matrix, carried)
    trace = np.clip((np.trace(carried_error, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    orientation_residual_deg = np.degrees(np.arccos(trace))

    result = {
        "motion_parquet": str(args.motion_parquet),
        "episode": args.episode,
        "tier": args.tier,
        "frames_compared": int(visible.sum()),
        "frames_invisible_skipped": int((~visible).sum()),
        "schema_version": motion.get("schema_version"),
        "motion_kind": motion.get("motion_kind"),
        "coord_frame": motion.get("coord_frame"),
        "object_name": motion.get("object_name"),
        "robot_joints": len(motion.get("robot_joint_names") or []),
        "ee_link_names": list(motion.get("ee_link_names") or []),
        "rigid_fit_residual_mm": round(residual_m * 1000.0, 4),
        "rigid_fit_rotation_deg": round(
            float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))), 4
        ),
        "rigid_fit_translation_m": [round(float(v), 5) for v in translation],
        "relative_rotation_max_diff_deg": round(float(angle_difference.max()), 5),
        "relative_rotation_mean_diff_deg": round(float(angle_difference.mean()), 6),
        "orientation_residual_mean_deg": round(float(orientation_residual_deg.mean()), 5),
        "orientation_residual_max_deg": round(float(orientation_residual_deg.max()), 5),
    }
    # A faithful retarget: one transform explains everything, to numerical precision.
    result["pass_rigid"] = result["rigid_fit_residual_mm"] < 1.0
    result["pass_orientation"] = result["orientation_residual_max_deg"] < 1.0
    result["pass"] = bool(result["pass_rigid"] and result["pass_orientation"])

    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
