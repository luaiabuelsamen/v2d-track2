"""Clean a Tier 3 object reconstruction using nothing but the reconstruction itself.

The diagnosis (`results/t3_recon_diag.json`) splits the 30 reconstructions three ways: 8 track
cleanly, 11 drift away slowly, and 11 contain *jumps* -- single frames or short runs where
FoundationPose relocalises onto the wrong hypothesis and then comes back. e01 is the clearest case:
10 frames jump more than 20 cm, yet its median error is 6.7 cm and its final frame is 1.5 cm from
truth. A jump like that is not a tracking failure, it is an outlier, and removing it is free.

Two passes, both causal-agnostic and both reading only the predicted trajectory:

1. **Jump rejection.** A frame is an outlier when the step into it and the step out of it are both
   large and roughly opposite -- the signature of a there-and-back relocalisation, as opposed to
   genuine fast motion, which is large and *consistent* in direction. Outlier runs up to
   `max_run` frames long are replaced by linear interpolation in position and slerp in rotation
   between the surviving neighbours. Longer runs are left alone: a 200-frame excursion is the
   tracker having genuinely lost the object, and inventing a path through it would be fabrication.

2. **Light smoothing.** A centred moving average of width `smooth` on position only. Orientation is
   left alone because averaging quaternions through a large rotation is worse than not.

Nothing here touches ground truth, and the output is still "our own reconstruction" -- the thing
Tier 3 asks us to bring. Scores before and after are reported per clip so the filter can be
rejected on the clips it does not help.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def find_outliers(
    position: np.ndarray, *, jump_cm: float, reversal: float, max_run: int
) -> np.ndarray:
    """Boolean mask of frames that look like a there-and-back relocalisation."""
    n = len(position)
    bad = np.zeros(n, dtype=bool)
    if n < 3:
        return bad
    step = np.diff(position, axis=0) * 100.0  # cm, step[i] = i -> i+1
    size = np.linalg.norm(step, axis=1)
    big = size > jump_cm
    # A run [a, b) is an outlier run when the step INTO a and the step OUT OF b-1 are both big and
    # point in roughly opposite directions, i.e. the trajectory leaves and returns.
    i = 0
    while i < len(big):
        if not big[i]:
            i += 1
            continue
        j = i + 1  # first index after the run of outlier frames, which starts at i+1
        while j < len(big) and not big[j] and j - i <= max_run:
            j += 1
        if j < len(big) and big[j] and (j - i) <= max_run:
            out, back = step[i], step[j]
            cosine = float(out @ back) / (np.linalg.norm(out) * np.linalg.norm(back) + 1e-12)
            if cosine < -reversal:
                bad[i + 1 : j + 1] = True
                i = j + 1
                continue
        i += 1
    bad[0] = bad[-1] = False  # the ends have no pair of neighbours to interpolate between
    return bad


def interpolate(
    position: np.ndarray, quat_xyzw: np.ndarray, bad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Replace masked frames by linear interpolation in position and slerp in rotation."""
    if not bad.any():
        return position, quat_xyzw
    good = np.where(~bad)[0]
    frames = np.arange(len(position))
    out_p = position.copy()
    for axis in range(3):
        out_p[bad, axis] = np.interp(frames[bad], good, position[good, axis])
    rotations = Rotation.from_quat(quat_xyzw[good])
    out_q = quat_xyzw.copy()
    out_q[bad] = Slerp(good, rotations)(np.clip(frames[bad], good[0], good[-1])).as_quat()
    return out_p, out_q


def moving_average(position: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average, with the window shrinking at the ends rather than padding."""
    if width <= 1:
        return position
    n = len(position)
    half = width // 2
    out = np.empty_like(position)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = position[lo:hi].mean(axis=0)
    return out


def clean_episode(
    path: Path,
    out_dir: Path,
    *,
    jump_cm: float,
    reversal: float,
    max_run: int,
    smooth: int,
) -> dict:
    blob = np.load(path, allow_pickle=True)
    poses = blob["pose_xyzw"]
    flat = poses.reshape(len(poses), -1)[:, :7]
    position, quat = flat[:, :3].copy(), flat[:, 3:7].copy()

    bad = find_outliers(position, jump_cm=jump_cm, reversal=reversal, max_run=max_run)
    position, quat = interpolate(position, quat, bad)
    smoothed = moving_average(position, smooth)

    out = np.concatenate([smoothed, quat], axis=1)
    episode = int(blob["episode_index"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / path.name,
        pose_xyzw=out.reshape(len(out), 1, 7),
        coordinate_frame=str(blob["coordinate_frame"]),
        episode_index=np.int64(episode),
        frame_index=np.asarray(blob["frame_index"], dtype=np.int64),
    )
    moved = np.linalg.norm(smoothed - flat[:, :3], axis=1) * 100.0
    edges = np.diff(np.concatenate([[0], bad.view(np.int8), [0]]))
    runs = edges.clip(min=0).sum()
    return {
        "episode": episode,
        "frames": len(out),
        "outlier_frames": int(bad.sum()),
        "outlier_runs": int(runs),
        "max_shift_cm": round(float(moved.max()), 3),
        "mean_shift_cm": round(float(moved.mean()), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--npz-in", type=Path, required=True, help="episode_%06d.npz, world frame")
    parser.add_argument("--npz-out", type=Path, required=True)
    parser.add_argument("--jump-cm", type=float, default=20.0, help="step size counting as a jump")
    parser.add_argument("--reversal", type=float, default=0.5, help="how opposite the return is")
    parser.add_argument("--max-run", type=int, default=12, help="longest run that may be bridged")
    parser.add_argument("--smooth", type=int, default=5, help="moving-average width on position")
    parser.add_argument("-o", "--out", type=Path, help="write the per-episode record here")
    args = parser.parse_args(argv)

    files = sorted(args.npz_in.glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"FATAL: no episode_*.npz in {args.npz_in}")
    records = [
        clean_episode(
            f,
            args.npz_out,
            jump_cm=args.jump_cm,
            reversal=args.reversal,
            max_run=args.max_run,
            smooth=args.smooth,
        )
        for f in files
    ]
    summary = {
        "what": "Tier 3 reconstruction cleaned with no ground truth: jumps rejected, then smoothed",
        "settings": {
            "jump_cm": args.jump_cm,
            "reversal": args.reversal,
            "max_run": args.max_run,
            "smooth": args.smooth,
        },
        "episodes_cleaned": len(records),
        "episodes_with_outliers": sum(1 for r in records if r["outlier_frames"]),
        "total_outlier_frames": sum(r["outlier_frames"] for r in records),
        "episodes": records,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
