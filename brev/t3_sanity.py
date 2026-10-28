"""Is a Tier 3 reconstruction actually tracking the object, before 29 more are paid for?

FoundationPose writes one `Transform3d` JSON per frame into `poses_dir`, in the CAMERA frame, so
the obvious check -- distance to the Tier 1 truth -- needs the frame-0 alignment that happens
later. These checks do not need it:

- **completeness**: a pose per video frame, and every entry finite. A silently short run is the
  failure that looks like success.
- **frozen**: the per-frame translation step. A tracker that lost the object and returned its
  initialisation forever has a median step of ~0.
- **jumping**: the largest step, against the reference's own largest step. Teleports mean
  re-registration, not tracking.
- **speed-profile correlation with the truth** -- the useful one, and it is alignment-free.
  A rigid camera-to-world transform rotates and translates the whole track, which leaves every
  |Δp| unchanged. So the reconstruction's speed over time can be compared with the released
  trajectory's speed over time directly, with no alignment and no fitting. A tracker that is
  following the object moves when the object moves; Spearman near zero means it is not.

    python brev/t3_sanity.py --poses DIR --episode 0 --dataset-root DATA [--tier tier1]

Prints one line per check and exits non-zero if completeness or finiteness fails, which are the
two that make a reconstruction unusable rather than merely poor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from v2d.dataset import object_pose  # noqa: E402


def read_poses(poses_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(translations (T,3), rotations as quaternions (T,4), frame indices)."""
    files = sorted(poses_dir.glob("*.json"), key=lambda p: int(p.stem))
    translations, rotations, frames = [], [], []
    for path in files:
        blob = json.loads(path.read_text())
        matrix = blob.get("matrix") or blob.get("transform") or blob.get("T")
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
            translations.append(matrix[:3, 3])
            rotations.append(Rotation.from_matrix(matrix[:3, :3]).as_quat())
        else:
            position = blob.get("translation") or blob.get("position")
            quaternion = blob.get("quaternion") or blob.get("rotation")
            translations.append(np.asarray(position, dtype=float).reshape(3))
            rotations.append(np.asarray(quaternion, dtype=float).reshape(4))
        frames.append(int(path.stem))
    return np.asarray(translations), np.asarray(rotations), frames


def speed(track: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(track, axis=0), axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tier", default="tier1", choices=("tier1", "tier2"))
    parser.add_argument("--json", dest="record", type=Path)
    args = parser.parse_args()

    positions, quaternions, frames = read_poses(args.poses)
    if len(positions) == 0:
        print("FAIL: no pose files at all")
        return 1
    truth, _visible = object_pose(args.dataset_root, args.tier, args.episode)
    expected = len(truth)

    finite = bool(np.isfinite(positions).all() and np.isfinite(quaternions).all())
    gaps = [b - a for a, b in zip(frames, frames[1:], strict=False) if b - a != 1]
    moved = speed(positions)
    truth_moved = speed(truth[:, :3])
    overlap = min(len(moved), len(truth_moved))
    rho, pvalue = (float("nan"), float("nan"))
    if overlap > 10:
        rho, pvalue = spearmanr(moved[:overlap], truth_moved[:overlap])

    angles = np.degrees(
        2 * np.arccos(np.clip(np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1)), 0, 1))
    )

    record = {
        "episode": args.episode,
        "poses": str(args.poses),
        "frames_written": len(positions),
        "frames_expected": expected,
        "complete": len(positions) == expected,
        "frame_index_gaps": gaps[:5],
        "all_finite": finite,
        "step_cm_median": float(np.median(moved) * 100) if len(moved) else None,
        "step_cm_p99": float(np.percentile(moved, 99) * 100) if len(moved) else None,
        "step_cm_max": float(moved.max() * 100) if len(moved) else None,
        "truth_step_cm_median": float(np.median(truth_moved) * 100),
        "truth_step_cm_max": float(truth_moved.max() * 100),
        "rotation_step_deg_median": float(np.median(angles)) if len(angles) else None,
        "rotation_step_deg_max": float(angles.max()) if len(angles) else None,
        "speed_spearman_vs_truth": None if np.isnan(rho) else float(rho),
        "speed_spearman_p": None if np.isnan(pvalue) else float(pvalue),
    }

    print(f"frames      {record['frames_written']}/{expected}"
          f"{'' if record['complete'] else '   <-- SHORT'}")
    print(f"finite      {finite}")
    print(f"gaps        {gaps[:5] if gaps else 'none'}")
    print(f"step cm     median {record['step_cm_median']:.3f}  p99 {record['step_cm_p99']:.2f}  "
          f"max {record['step_cm_max']:.2f}   (truth: median "
          f"{record['truth_step_cm_median']:.3f}, max {record['truth_step_cm_max']:.2f})")
    print(f"rot step    median {record['rotation_step_deg_median']:.3f} deg  "
          f"max {record['rotation_step_deg_max']:.2f} deg")
    print(f"speed rho   {record['speed_spearman_vs_truth']}  (p={record['speed_spearman_p']})")

    verdict = []
    if not record["complete"]:
        verdict.append("INCOMPLETE")
    if not finite:
        verdict.append("NOT FINITE")
    if record["step_cm_median"] is not None and record["step_cm_median"] < 1e-4:
        verdict.append("FROZEN (median step under 1 micron)")
    if (record["step_cm_max"] or 0) > 20 * max(record["truth_step_cm_max"], 1e-6):
        verdict.append("JUMPING (a step 20x the reference's largest)")
    if record["speed_spearman_vs_truth"] is not None and record["speed_spearman_vs_truth"] < 0.1:
        verdict.append("UNCORRELATED with the reference's motion")
    print("verdict     " + (", ".join(verdict) if verdict else "looks like a track"))
    record["verdict"] = verdict or ["looks like a track"]

    if args.record:
        args.record.write_text(json.dumps(record, indent=1) + "\n")
    return 1 if ("INCOMPLETE" in verdict or "NOT FINITE" in verdict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
