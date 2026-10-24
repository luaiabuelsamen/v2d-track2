"""Pack a Track 2 Tier 1 submission with the kit's own packer, and refuse to fake one.

The Tier 1 entry is the insurance policy: a valid submission to any track before the freeze is
what keeps the Brev credit. This script is that path, built and proven end to end, and it is
deliberately unable to produce a submittable CSV out of what we have today.

WHY IT REFUSES
  The kit asks for "the rolled-out object trajectory" -- what a trained policy achieved in
  simulation -- together with the evaluator's script and evaluation-code SHA-256 and the policy
  checkpoint's SHA-256. What this repo can produce at $0 is a *kinematic* retarget: the object is
  carried by the retargeted G1's hands with no physics. That trajectory is a rigid transform of
  the released input to **0.0 mm over every one of 22,989 frames** (`results/retarget_all.json`),
  so after the packer's frame conversion a Tier 1 submission built from it would be the scoring
  reference echoed back, scoring ~1.0 on every metric, under provenance fields we would have to
  invent. That is not a weak entry, it is a misrepresentation, and the guard below exists so it
  cannot be uploaded by accident.

  `--dry-run` therefore stamps its output `NOT_FOR_SUBMISSION` in the row ids' provenance columns
  and the file name, proving the machinery while making the artefact unusable as an entry. A
  submittable CSV needs `--rollout-root` pointing at a real FlashCHORD export: schema
  `flash_chord_object_trajectories_v1`, carrying both achieved and reference poses and all three
  genuine hashes. One Brev gate run produces that for one clip; 30 produce the entry.

WHAT IT DOES
  1. Reads the retargeted motion parquets (`object_body_position`, `object_body_wxyz`) or a real
     rollout export.
  2. For the kinematic path, fits the rigid transform taking the retargeted object trajectory
     onto the released Tier 1 input (Kabsch over the visible frames, the same fit
     `validate_retarget.py` reports) and applies it, so the poses land in
     `released_track2_world`. The residual is printed, because a conversion you cannot measure is
     a conversion you should not trust.
  3. Writes one `episode_%06d.parquet` per episode in the kit's accepted legacy layout.
  4. Invokes `tools/pack_submission.py` from the kit -- the organisers' code, not a
     reimplementation -- so the row grid, episode roster and object order are validated by the
     same script that will validate the real entry.
  5. Checks the CSV's row ids against `data/track_2_sample_submission.csv` exactly, and scores it
     locally with `scripts/score_track2.py --pred csv:...`.

It never uploads, never registers and never touches the network.

Usage:
  python scripts/pack_tier1_submission.py --kit-root KIT --dataset-root DATA \
      --motion-root MOTION/tier1 --work-dir DIR --dry-run [-o results/tier1_package.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TIER1 = "tier_1_multiview_caption"
STAMP = "NOT_FOR_SUBMISSION"
#: Matches the packer's GitHub-commit pattern while being transparently not a real commit.
DRY_RUN_COMMIT = "https://github.com/NOT-FOR-SUBMISSION/v2d-dry-run/commit/" + "0" * 40
LEGACY_COLUMNS = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rigid (R, t) minimising |R source + t - target|, with a proper rotation.

    Duplicated from scripts/validate_retarget.py rather than imported: these tools run under
    `python -I`, which implies `-P`, so the script's own directory is not on sys.path and
    scripts/ cannot import from itself.
    """
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, target_mean - rotation @ source_mean


def rotate_poses(pose: np.ndarray, rotation: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Apply a rigid (rotation, offset) to (T,7) xyz+xyzw poses."""
    from scipy.spatial.transform import Rotation

    out = np.empty_like(pose)
    out[:, :3] = pose[:, :3] @ rotation.T + offset
    out[:, 3:] = Rotation.from_matrix(
        rotation @ Rotation.from_quat(pose[:, 3:]).as_matrix()
    ).as_quat()
    return out


def released_input(dataset_root: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """(T,7) xyz+xyzw Tier 1 object pose and its visibility mask."""
    table = pq.read_table(
        dataset_root / "track_2" / TIER1 / "data" / "chunk-000" / f"episode_{episode:06d}.parquet",
        columns=["observation.object.pose", "observation.object.visible"],
    )
    pose = np.asarray(table["observation.object.pose"].to_pylist(), dtype=np.float64)
    visible = np.asarray(table["observation.object.visible"].to_pylist(), dtype=bool)
    # released order is xyz + wxyz
    return np.concatenate([pose[:, :3], pose[:, [4, 5, 6, 3]]], axis=1), visible


def retargeted_object(motion_parquet: Path) -> np.ndarray:
    """(T,7) xyz+xyzw object pose out of a motion_v1 single_robot parquet."""
    table = pq.read_table(motion_parquet, columns=["object_body_position", "object_body_wxyz"])
    position = np.asarray(table["object_body_position"].to_pylist()[0], dtype=np.float64)
    wxyz = np.asarray(table["object_body_wxyz"].to_pylist()[0], dtype=np.float64)
    if position.ndim == 3:  # (T,1,3) single body
        position, wxyz = position[:, 0], wxyz[:, 0]
    return np.concatenate([position, wxyz[:, [1, 2, 3, 0]]], axis=1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kit-root", type=Path, required=True, help="extracted v2d_submission_kit/"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--motion-root", type=Path, help="retargeted tier1 motion tree; the kinematic path"
    )
    parser.add_argument(
        "--rollout-root", type=Path, help="a real FlashCHORD export; the only submittable path"
    )
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="scratch for the staged parquets"
    )
    parser.add_argument("--code-commit", default="", help="GitHub /commit/<sha> URL for provenance")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stamp the output NOT_FOR_SUBMISSION and proceed without provenance",
    )
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    if (args.motion_root is None) == (args.rollout_root is None):
        raise SystemExit("pass exactly one of --motion-root (kinematic) or --rollout-root (real)")
    if args.motion_root is not None and not args.dry_run:
        raise SystemExit(
            "refusing to pack a submittable CSV from the kinematic retarget.\n"
            "  The retargeted object trajectory is a rigid transform of the released input to\n"
            "  0.0 mm, so a Tier 1 entry built from it is the scoring reference echoed back, and\n"
            "  the evaluator/checkpoint hashes the kit asks for do not exist for it.\n"
            "  Use --dry-run to prove the packaging path, or --rollout-root for a real export."
        )

    kit = args.kit_root
    sample = kit / "data" / "track_2_sample_submission.csv"
    metadata = [
        json.loads(line)
        for line in (args.dataset_root / "track_2" / TIER1 / "meta" / "episodes_metadata.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    metadata.sort(key=lambda r: r["episode_index"])

    results_dir = args.work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    conversions = []

    if args.rollout_root is not None:
        results_dir = args.rollout_root
        print(f"using the real rollout export at {results_dir}")
    else:
        for row in metadata:
            episode = int(row["episode_index"])
            motion = (
                args.motion_root
                / "whole_body"
                / "soma"
                / f"sequence_id={row['sequence_id']}"
                / "robot_name=g1"
                / "data.parquet"
            )
            if not motion.exists():
                raise SystemExit(f"missing retargeted motion for e{episode:02d}: {motion}")
            achieved = retargeted_object(motion)
            target, visible = released_input(args.dataset_root, episode)
            frames = min(len(achieved), len(target))
            achieved, target, visible = achieved[:frames], target[:frames], visible[:frames]
            # Fit on the visible frames only: the one invisible frame in the release is
            # zero-filled, and a zero quaternion reads back as identity at the world origin.
            rotation, offset = kabsch(achieved[visible, :3], target[visible, :3])
            converted = rotate_poses(achieved, rotation, offset)
            residual = float(
                np.linalg.norm(converted[visible, :3] - target[visible, :3], axis=1).max()
            )
            conversions.append(
                {
                    "episode": episode,
                    "frames": frames,
                    "fit_residual_max_mm": round(residual * 1000, 6),
                    "frames_invisible": int((~visible).sum()),
                }
            )
            table = pa.table(
                {
                    name: converted[:, index].astype(np.float64)
                    for index, name in enumerate(LEGACY_COLUMNS)
                }
                | {
                    "frame_index": np.arange(frames, dtype=np.int64),
                    "world_index": np.zeros(frames, dtype=np.int64),
                    "object_slot": np.zeros(frames, dtype=np.int64),
                }
            )
            # Legacy layout on purpose: no evaluator reference, because there is no evaluator.
            # eval_code_sha256 is required by the packer and we have nothing true to put there,
            # so it carries the stamp rather than a plausible-looking hash. It is not 64 hex
            # characters, it cannot be mistaken for a digest, and the packer copies it into all
            # 22,989 rows, so the artefact labels itself whatever the file is renamed to.
            # checkpoint_sha256 gets the stamp too. The packer allows the set to be empty at
            # its line 437 and then rejects any blank provenance cell at 488, so a real
            # submission must record one per episode; leaving it empty also trips a bug in the
            # packer's own error path (`row.split("_")[1]` on `track_2/e000000/...` raises
            # ValueError instead of printing the message).
            table = table.replace_schema_metadata(
                {
                    "episode_index": str(episode),
                    "eval_code_sha256": STAMP,
                    "checkpoint_sha256": STAMP,
                }
            )
            pq.write_table(table, results_dir / f"episode_{episode:06d}.parquet")
        worst = max(c["fit_residual_max_mm"] for c in conversions)
        print(
            f"staged {len(conversions)} episodes; worst frame-to-released residual {worst:.6f} mm"
        )
        if worst < 0.001:
            print(
                "  -> the retargeted object IS the released input to within a micrometre. "
                "This confirms why this path is not a submittable entry."
            )

    out_csv = args.work_dir / (f"track2_tier1_{STAMP}.csv" if args.dry_run else "track2_tier1.csv")
    command = [
        sys.executable,
        str(kit / "tools" / "pack_submission.py"),
        "--track",
        "track_2_tier1",
        "--results",
        str(results_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--sample",
        str(sample),
        "--code-commit",
        args.code_commit or DRY_RUN_COMMIT,
        "-o",
        str(out_csv),
    ]
    if args.motion_root is not None:
        # The packer needs an eval_script_sha256 and will hash a script for us. Hashing the
        # kit's own released evaluator is honest only because the packer prints, and this
        # script records, that we did not run it.
        command += ["--eval-script", str(kit / "flash_chord" / "scripts" / "evaluate_policy.py")]
    print("\n$ " + " ".join(command))
    packed = subprocess.run(
        command,
        cwd=str(kit),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(kit)},
    )
    print(packed.stdout[-4000:] or "(no stdout)")
    if packed.returncode != 0:
        print(packed.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"the kit's packer exited {packed.returncode}")

    import pandas as pd

    submitted = pd.read_csv(out_csv)
    expected = pd.read_csv(sample)
    rows_match = list(submitted["row_id"]) == list(expected["row_id"])
    print(f"\nrows: {len(submitted)} (sample {len(expected)}); row ids identical: {rows_match}")
    if not rows_match:
        raise SystemExit("row ids do not match the kit's sample submission")

    score_json = args.work_dir / "local_score.json"
    score = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/score_track2.py",
            "--kit-root",
            str(kit),
            "--dataset-root",
            str(args.dataset_root),
            "--tier",
            "tier1",
            "--pred",
            f"csv:{out_csv}",
            "-o",
            str(score_json),
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    print("\nlocal score of the packed CSV:")
    print(score.stdout or score.stderr[-2000:])
    scored = json.loads(score_json.read_text())["overall"] if score_json.exists() else None

    result = {
        "what": "Track 2 Tier 1 submission package, built with the kit's own packer",
        "submittable": bool(args.rollout_root is not None and not args.dry_run),
        "stamped": STAMP if args.dry_run else None,
        "source": "kinematic retarget" if args.motion_root else "flashchord rollout",
        "csv": str(out_csv),
        "rows": len(submitted),
        "row_ids_match_sample": rows_match,
        "packer": "kit tools/pack_submission.py",
        "frame_conversion": conversions,
        "local_score": scored,
        "why_not_submittable": None
        if args.rollout_root
        else (
            "the kinematic retarget reproduces the released input to 0.0 mm, so this CSV is the "
            "scoring reference echoed back; eval_code_sha256 carries the literal string "
            f"{STAMP!r} instead of a digest, so does checkpoint_sha256, and eval_script_sha256 "
            "is the hash of the kit's released evaluator, which was never run"
        ),
        "provenance_gaps": None
        if args.rollout_root
        else {
            "eval_script_sha256": "hash of the kit's released evaluate_policy.py; NOT run here",
            "eval_code_sha256": STAMP,
            "checkpoint_sha256": STAMP + ": no policy was trained",
            "code_commit_url": DRY_RUN_COMMIT,
        },
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
