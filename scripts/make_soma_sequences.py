"""Convert released Track 2 episodes into the sequence folders `soma_to_g1.py` expects.

The challenge publishes each episode as one LeRobot parquet. The official retargeter wants a
directory per sequence instead:

    sequence_id=<id>/
        soma_params.npz                        SOMA-X export fields
        poses.npy                              (T,4,4) object world transforms, OpenCV world
        ground_plane.json                      copied from the release
        reconstructed_mesh/output_aligned.glb  the object mesh

Field contracts, all read out of `robotic_grounding/retarget/read_soma.py` rather than guessed:

- `poses` is `(T,J,3)` **local** rotation vectors and `keep_root=False` means `poses[:,0]` is
  the root, which matches the released 231 = 77x3 layout.
- `joint_names` must equal `SOMALayer.rig_data["joint_names"][1:]` exactly, so this script
  reads them straight out of `SOMA_neutral.npz` and drops the leading "Root" entry. Of the 78
  names there, 77 remain, which is the released joint count.
- `identity_coeffs` and `scale_params` are indexed `[0]`, so they are stored with a leading
  axis of 1. Both are constant over time in every released episode; the script checks that.
- `unit` is "meters" and `absolute_pose` is False.
- `joint_orient` is optional: when absent, `read_soma` derives it from the active model's
  t-pose, which is what the docstring calls authoritative. We omit it.
- `poses.npy` is consumed as `einsum("ij,tjk->tik", diag(1,-1,-1,1), poses)`, so it must hold
  4x4 homogeneous matrices, not 7-vectors.

Usage:
  python scripts/make_soma_sequences.py --dataset-root DATA --tier tier1 --out DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TIER_DIRS = {"tier1": "tier_1_multiview_caption", "tier2": "tier_2_synthetic_noise"}
EXPECTED_JOINTS = 77


def rig_joint_names(soma_neutral: Path) -> list[str]:
    """The J joint names `read_soma` validates against, i.e. the rig list without "Root"."""
    with np.load(soma_neutral, allow_pickle=True) as archive:
        names = [str(n) for n in archive["joint_names"]]
    if names[0].lower() != "root":
        raise SystemExit(f"expected a leading Root entry in {soma_neutral}, got {names[0]!r}")
    return names[1:]


def pose_matrices(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(T,3) + (T,4) wxyz -> (T,4,4) homogeneous transforms."""
    w, x, y, z = (quat_wxyz[:, i] for i in range(4))
    norm = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("degenerate object quaternion")
    w, x, y, z = (c / norm[:, 0] for c in (w, x, y, z))
    rotation = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    out = np.zeros((len(position), 4, 4), dtype=np.float64)
    out[:, :3, :3] = rotation
    out[:, :3, 3] = position
    out[:, 3, 3] = 1.0
    return out


def hold_last_visible(pose: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, int]:
    """Carry the last visible object pose over invisible frames.

    An invisible frame is zero-filled, and a zero quaternion is not a rotation. Exactly one
    frame in the whole release is affected -- episode 11's last frame -- and it is also the one
    frame the organizers leave out of the scored grid, which is presumably why.
    """
    if visible.all():
        return pose, 0
    index = np.arange(len(visible))
    index[~visible] = -1
    index = np.maximum.accumulate(index)
    if index[0] < 0:
        index[index < 0] = int(np.argmax(visible)) if visible.any() else 0
    return pose[index], int((~visible).sum())


def mesh_source(mesh_root: Path, name: str) -> Path:
    """The .glb to hand the retargeter, preferring the visual variant like the kit does."""
    for candidate in (f"{name}_visual.glb", f"{name}.glb"):
        path = mesh_root / name / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"no mesh for {name!r} under {mesh_root / name}")


def convert(dataset_root: Path, tier: str, out_root: Path, soma_neutral: Path, episodes: list[int] | None) -> list[dict]:
    tier_root = dataset_root / "track_2" / TIER_DIRS[tier]
    clean_root = dataset_root / "track_2" / TIER_DIRS["tier1"]  # meshes and planes ship per tier, identically
    metadata = [json.loads(l) for l in (tier_root / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()]
    metadata.sort(key=lambda r: r["episode_index"])
    names = rig_joint_names(soma_neutral)
    if len(names) != EXPECTED_JOINTS:
        raise SystemExit(f"rig has {len(names)} joints after dropping Root, expected {EXPECTED_JOINTS}")

    written = []
    for row in metadata:
        episode = int(row["episode_index"])
        if episodes is not None and episode not in episodes:
            continue
        table = pq.read_table(tier_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
        pose = np.asarray(table["observation.human.pose"].to_pylist(), dtype=np.float64)
        transl = np.asarray(table["observation.human.translation"].to_pylist(), dtype=np.float64)
        identity = np.asarray(table["observation.human.identity_coeffs"].to_pylist(), dtype=np.float64)
        scale = np.asarray(table["observation.human.scale_params"].to_pylist(), dtype=np.float64)
        flexibles = np.asarray(table["observation.human.bone_length_flexibles"].to_pylist(), dtype=np.float64)
        object_pose = np.asarray(table["observation.object.pose"].to_pylist(), dtype=np.float64)
        visible = np.asarray(table["observation.object.visible"].to_pylist(), dtype=bool)
        object_pose, filled = hold_last_visible(object_pose, visible)

        frames = len(pose)
        if pose.shape[1] != EXPECTED_JOINTS * 3:
            raise SystemExit(f"episode {episode}: pose is {pose.shape}, expected (T,{EXPECTED_JOINTS * 3})")
        for label, array in (("identity_coeffs", identity), ("scale_params", scale)):
            if not np.allclose(array, array[0]):
                raise SystemExit(f"episode {episode}: {label} varies over time; read_soma takes only frame 0")

        sequence = out_root / f"sequence_id={row['sequence_id']}"
        (sequence / "reconstructed_mesh").mkdir(parents=True, exist_ok=True)
        np.savez(
            sequence / "soma_params.npz",
            poses=pose.reshape(frames, EXPECTED_JOINTS, 3).astype(np.float32),
            transl=transl.astype(np.float32),
            joint_names=np.array(names, dtype="<U64"),
            identity_model_type=np.array("mhr"),
            identity_coeffs=identity[:1].astype(np.float32),
            scale_params=scale[:1].astype(np.float32),
            bone_length_flexibles=flexibles[:1].astype(np.float32),
            unit=np.array("meters"),
            keep_root=np.array(False),
            absolute_pose=np.array(False),
            rotation_repr=np.array("rotvec"),
        )
        # released order is xyz + wxyz
        np.save(sequence / "poses.npy", pose_matrices(object_pose[:, :3], object_pose[:, 3:]))
        shutil.copy2(clean_root / row["ground_plane"], sequence / "ground_plane.json")
        source = mesh_source(clean_root / "mesh", row["object"])
        destination = sequence / "reconstructed_mesh" / "output_aligned.glb"
        if not destination.exists():
            shutil.copy2(source, destination)

        written.append(
            {"episode": episode, "sequence_id": row["sequence_id"], "object": row["object"],
             "frames": frames, "invisible_frames_filled": filled, "folder": str(sequence)}
        )
        note = f"  ({filled} invisible frame(s) held)" if filled else ""
        print(f"e{episode:02d} {row['object'][:22]:22s} {frames:5d} frames -> {sequence.name}{note}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tier", default="tier1", choices=sorted(TIER_DIRS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--soma-neutral", type=Path, required=True, help="staged SOMA_neutral.npz")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    written = convert(args.dataset_root, args.tier, args.out, args.soma_neutral, args.episodes)
    print(f"\n{len(written)} sequence folder(s) under {args.out}")


if __name__ == "__main__":
    main()
