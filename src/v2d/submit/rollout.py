"""Turn a FlashCHORD rollout into the npz the scorer reads, in the released frame.

The evaluator writes `achieved_*` and `reference_*` poses in **its own** frame, not the released
Track 2 world frame. Scoring the achieved pose directly is the single easiest way to get a
meaningless number: on a real rollout it read MPPE 306.74 cm instead of 14.74 cm, a 20x error
that looks like a catastrophic policy rather than a frame mismatch.

The conversion is the one the kit's packer performs, reproduced here so a rollout can be scored
without packing a full 30-episode submission: fit a single rigid transform from frame 0 of the
evaluator's recorded reference to frame 0 of the tier's released input, verify the whole
reference maps across within tolerance, then apply that same transform to the achieved pose.
The verification is the point -- it is what catches a frame or convention mismatch, and on a
correct rollout it comes out at 0.0000 cm.

Only world 0 is taken. Under `reset_mode=explicit` every world resets to the same frame and runs
a deterministic policy, so the cohort is redundant and the challenge submits one world.

**If the policy trained on a CLEANED reference, pass it with `--reference-npz`.** A cleaned
reference differs from the released tier input by a per-frame correction, not by one rigid
transform, so converting against the released input fits a transform that is right at frame 0
and drifts after it. The reference check catches this -- it is what it is for -- and the kit's
packer has the same escape hatch in `--frame-references`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from ..config import TIER_DIRS
from ..geometry import wxyz_to_xyzw

#: The packer's own tolerance for "these differ by one rigid transform".
POSITION_TOLERANCE_M = 1e-4
ANGLE_TOLERANCE_RAD = 1e-4


def _poses(table: Any, prefix: str, keep: np.ndarray, order: np.ndarray) -> np.ndarray:
    columns = [f"{prefix}_position_{axis}" for axis in "xyz"] + [
        f"{prefix}_quaternion_{axis}" for axis in "xyzw"
    ]
    return np.stack([table[column].to_numpy()[keep][order] for column in columns], axis=-1).astype(
        np.float64
    )


def released_pose(dataset_root: Path, tier: str, episode: int) -> np.ndarray:
    """The tier's released object pose as ``(T,7)`` xyz + xyzw."""
    path = (
        dataset_root
        / "track_2"
        / TIER_DIRS[tier]
        / "data"
        / "chunk-000"
        / f"episode_{episode:06d}.parquet"
    )
    raw = np.asarray(
        pq.read_table(path, columns=["observation.object.pose"])[
            "observation.object.pose"
        ].to_pylist(),
        dtype=np.float64,
    )
    return np.concatenate([raw[:, :3], wxyz_to_xyzw(raw[:, 3:])], axis=1)


def convert(
    achieved: np.ndarray, evaluator_reference: np.ndarray, released: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Map `achieved` into the released frame, and report how well the reference maps across."""
    rotation = (
        Rotation.from_quat(released[0, 3:]) * Rotation.from_quat(evaluator_reference[0, 3:]).inv()
    )
    offset = released[0, :3] - rotation.apply(evaluator_reference[0, :3])

    def apply(pose: np.ndarray) -> np.ndarray:
        out = np.empty_like(pose)
        out[:, :3] = rotation.apply(pose[:, :3]) + offset
        out[:, 3:] = (rotation * Rotation.from_quat(pose[:, 3:])).as_quat()
        return out

    checked = apply(evaluator_reference)
    position_residual = float(np.linalg.norm(checked[:, :3] - released[:, :3], axis=1).max())
    angle_residual = float(
        (Rotation.from_quat(checked[:, 3:]) * Rotation.from_quat(released[:, 3:]).inv())
        .magnitude()
        .max()
    )
    return apply(achieved), position_residual, angle_residual


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="v2d rollout-to-npz",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trajectory", type=Path, help="a flash_chord object-trajectories parquet")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--tier",
        default="tier1",
        choices=sorted(TIER_DIRS),
        help="the tier whose input the evaluator rolled out against",
    )
    parser.add_argument(
        "--reference-npz",
        type=Path,
        help="the CLEANED reference the policy actually trained on (episode_NNNNNN.npz in "
        "released_track2_world). Needed when the policy did not train on the raw released tier "
        "input: otherwise the conversion fits frame 0 and drifts after it.",
    )
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--npz-out", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path, help="also write a JSON record")
    args = parser.parse_args(argv)

    table = pq.read_table(args.trajectory)
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    keep = (table["environment_id"].to_numpy() == args.world) & (
        table["object_body_id"].to_numpy() == 0
    )
    order = np.argsort(table["reference_frame"].to_numpy()[keep])
    frames = table["reference_frame"].to_numpy()[keep][order]

    achieved = _poses(table, "achieved", keep, order)
    evaluator_reference = _poses(table, "reference", keep, order)
    if args.reference_npz is not None:
        with np.load(args.reference_npz / f"episode_{args.episode:06d}.npz") as archive:
            if str(archive["coordinate_frame"]) != "released_track2_world":
                raise SystemExit(
                    f"{args.reference_npz}: coordinate_frame must be released_track2_world"
                )
            target = np.asarray(archive["pose_xyzw"], dtype=np.float64).reshape(-1, 7)
        released = target[frames]
        reference_source = str(args.reference_npz)
    else:
        released = released_pose(args.dataset_root, args.tier, args.episode)[frames]
        reference_source = f"released {args.tier} input"

    converted, position_residual, angle_residual = convert(achieved, evaluator_reference, released)
    within = position_residual <= POSITION_TOLERANCE_M and angle_residual <= ANGLE_TOLERANCE_RAD

    args.npz_out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.npz_out / f"episode_{args.episode:06d}.npz",
        episode_index=args.episode,
        coordinate_frame="released_track2_world",
        frame_index=frames.astype(np.int64),
        pose_xyzw=converted.reshape(len(converted), 1, 7),
    )

    record = {
        "what": "a FlashCHORD rollout converted into the released Track 2 world frame",
        "trajectory": str(args.trajectory),
        "episode": args.episode,
        "tier_rolled_out_against": args.tier,
        "reference_converted_against": reference_source,
        "world": args.world,
        "frames": len(frames),
        "schema": metadata.get("flash_chord.schema"),
        "checkpoint_sha256": metadata.get("flash_chord.checkpoint_sha256"),
        "eval_script_sha256": metadata.get("flash_chord.eval_script_sha256"),
        "eval_code_sha256": metadata.get("flash_chord.eval_code_sha256"),
        "reference_check_position_cm": round(position_residual * 100, 6),
        "reference_check_angle_deg": round(np.degrees(angle_residual), 6),
        "reference_check_within_packer_tolerance": within,
        "achieved_vs_released_mean_cm": round(
            float(np.linalg.norm(converted[:, :3] - released[:, :3], axis=1).mean() * 100), 4
        ),
        "npz_out": str(args.npz_out),
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    if not within:
        print(
            "WARNING: the evaluator reference does not map onto the released input within the "
            "packer's tolerance. The achieved poses are converted anyway, but a frame or "
            "convention mismatch is the likeliest cause and the score will be meaningless.",
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
