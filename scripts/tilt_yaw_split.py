"""Split the Tier 2 orientation error into the part support contact could fix and the part it cannot.

An object resting on a surface has two of its three rotational degrees of freedom pinned by
that contact: the direction of the support normal in the object's own frame. The third, a
rotation about the normal, is free. So a reference cleaner built on support contact and
non-penetration can only ever fix the first part.

For every frame this script writes the plane normal into the object frame,

    m = R^T n        (n = ground-plane normal, R = object->world rotation)

and calls angle(m_tier2, m_clean) the **tilt** error and whatever is left after aligning the
two the **yaw** error. It then emits two hypothetical trajectories:

    tilt_fixed   Tier 2 orientation with its tilt replaced by the clean tilt, Tier 2 yaw kept
    yaw_fixed    Tier 2 orientation with its yaw replaced by the clean yaw, Tier 2 tilt kept
    rot_fixed    Tier 2 position with the clean orientation outright

All three keep the Tier 2 position, so `rot_fixed` is the ceiling for a cleaner that fixes
orientation alone, and tilt_fixed/yaw_fixed decompose the way to it.

Scoring those with scripts/score_track2.py bounds what a support-contact cleaner can buy,
before any of it is built. The clean reference is used here only to measure recoverability;
it is not an input a submitted pipeline may use.

It also checks whether the released ground plane actually supports the clean trajectory, by
reporting how far the lowest mesh vertex sits from the plane while the object is at rest.

Usage:
  python scripts/tilt_yaw_split.py --dataset-root DATA --npz-out DIR -o results/tilt_yaw_split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh

TIER1, TIER2 = "tier_1_multiview_caption", "tier_2_synthetic_noise"
REST_SPEED_M_PER_S = 0.02  # object origin speed below this counts as "at rest"
MAX_PROBE_VERTICES = 20000


def rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (quat_xyzw[..., i] for i in range(4))
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*quat_xyzw.shape[:-1], 3, 3)


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,4) xyzw."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    return Rotation.from_matrix(matrix).as_quat()


def align_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(...,3,3) rotations taking unit vector `source` onto unit vector `target`."""
    axis = np.cross(source, target)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    dot = np.clip(np.sum(source * target, axis=-1), -1.0, 1.0)[..., None, None]
    identity = np.broadcast_to(np.eye(3), (*source.shape[:-1], 3, 3)).copy()
    safe = np.where(norm > 1e-9, norm, 1.0)
    unit = axis / safe
    zero = np.zeros(source.shape[:-1])
    cross = np.stack(
        [
            np.stack([zero, -unit[..., 2], unit[..., 1]], -1),
            np.stack([unit[..., 2], zero, -unit[..., 0]], -1),
            np.stack([-unit[..., 1], unit[..., 0], zero], -1),
        ],
        axis=-2,
    )
    sin = norm[..., None]
    out = identity + sin * cross + (1.0 - dot) * (cross @ cross)
    # antiparallel (dot == -1) has no unique axis; those frames keep identity and are rare
    return np.where(norm[..., None] > 1e-9, out, identity)


def geodesic_deg(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(a_xyzw * b_xyzw, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot))


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["observation.object.pose", "observation.object.visible"])
    pose = np.asarray(table["observation.object.pose"].to_pylist(), dtype=np.float64)
    visible = np.asarray(table["observation.object.visible"].to_pylist(), dtype=bool)
    pose = np.concatenate([pose[:, :3], pose[:, [4, 5, 6, 3]]], axis=1)
    if not visible.all():
        index = np.arange(len(visible))
        index[~visible] = -1
        index = np.maximum.accumulate(index)
        index[index < 0] = int(np.argmax(visible)) if visible.any() else 0
        pose = pose[index]
    return pose, visible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--npz-out", type=Path,
                        help="write tilt_fixed/, yaw_fixed/ and rot_fixed/ npz trajectories here")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    root = args.dataset_root / "track_2" / TIER1
    metadata = [json.loads(l) for l in (root / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()]
    metadata.sort(key=lambda r: r["episode_index"])

    episodes, meshes = {}, {}
    for row in metadata:
        episode = int(row["episode_index"])
        name = row["object"]
        clean, _ = load_pose(root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
        noisy, _ = load_pose(
            args.dataset_root / "track_2" / TIER2 / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        )
        plane = np.asarray(json.loads((root / row["ground_plane"]).read_text())["plane"], dtype=np.float64)

        normal = plane[:3] / np.linalg.norm(plane[:3])
        offset = plane[3] / np.linalg.norm(plane[:3])
        # Signed distance is normal.x + offset. The object must sit on the +side; pick the sign
        # of "up" that puts the clean trajectory above the plane.
        if np.mean(clean[:, :3] @ normal + offset) < 0:
            normal, offset = -normal, -offset

        if name not in meshes:
            mesh = trimesh.load(str(root / row["mesh"]), force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            if len(vertices) > MAX_PROBE_VERTICES:
                pick = np.random.default_rng(0).choice(len(vertices), MAX_PROBE_VERTICES, replace=False)
                vertices = vertices[pick]
            meshes[name] = vertices
        vertices = meshes[name]

        r_clean = rotation_matrix(clean[:, 3:])
        r_noisy = rotation_matrix(noisy[:, 3:])
        m_clean = np.einsum("tji,j->ti", r_clean, normal)
        m_noisy = np.einsum("tji,j->ti", r_noisy, normal)
        tilt = np.degrees(np.arccos(np.clip(np.sum(m_clean * m_noisy, axis=-1), -1.0, 1.0)))
        total = geodesic_deg(clean[:, 3:], noisy[:, 3:])

        # tilt_fixed: swap the object-frame support normal for the clean one, keep the yaw error
        correction = align_rotation(m_noisy, m_clean)  # acts in the object frame
        r_tilt = r_noisy @ np.swapaxes(correction, -1, -2)
        q_tilt = quat_from_matrix(r_tilt)
        yaw = geodesic_deg(clean[:, 3:], q_tilt)
        # yaw_fixed: the complement, so tilt+yaw arms bracket the total error
        r_yaw = r_clean @ correction
        q_yaw = quat_from_matrix(r_yaw)

        speed = np.linalg.norm(np.gradient(noisy[:, :3], axis=0), axis=-1) * 30.0
        rest = speed < REST_SPEED_M_PER_S

        lowest_clean = (np.einsum("tij,vj->tvi", r_clean, vertices) + clean[:, None, :3]) @ normal + offset
        lowest_clean = lowest_clean.min(axis=1)
        lowest_noisy = (np.einsum("tij,vj->tvi", r_noisy, vertices) + noisy[:, None, :3]) @ normal + offset
        lowest_noisy = lowest_noisy.min(axis=1)

        episodes[episode] = {
            "object": name,
            "frames": int(len(clean)),
            "rest_fraction": float(rest.mean()),
            "rot_err_total_deg": float(total.mean()),
            "rot_err_tilt_deg": float(tilt.mean()),
            "rot_err_yaw_deg": float(yaw.mean()),
            "rot_err_total_deg_rest": float(total[rest].mean()) if rest.any() else None,
            "rot_err_tilt_deg_rest": float(tilt[rest].mean()) if rest.any() else None,
            "rot_err_yaw_deg_rest": float(yaw[rest].mean()) if rest.any() else None,
            "clean_lowest_vertex_to_plane_cm": float(lowest_clean.mean() * 100.0),
            "clean_lowest_vertex_to_plane_cm_rest": float(lowest_clean[rest].mean() * 100.0) if rest.any() else None,
            "tier2_lowest_vertex_to_plane_cm_rest": float(lowest_noisy[rest].mean() * 100.0) if rest.any() else None,
            "mesh_extent_cm": float(np.ptp(vertices, axis=0).max() * 100.0),
        }

        if args.npz_out:
            for label, quat in (("tilt_fixed", q_tilt), ("yaw_fixed", q_yaw),
                                ("rot_fixed", clean[:, 3:])):
                directory = args.npz_out / label
                directory.mkdir(parents=True, exist_ok=True)
                # Both arms keep the Tier 2 position, so they differ only in orientation.
                pose = np.concatenate([noisy[:, :3], quat], axis=1)
                np.savez(
                    directory / f"episode_{episode:06d}.npz",
                    episode_index=episode,
                    coordinate_frame="released_track2_world",
                    frame_index=np.arange(len(pose)),
                    pose_xyzw=pose.reshape(len(pose), 1, 7),
                )

    def mean(key: str) -> float:
        values = [v[key] for v in episodes.values() if v[key] is not None]
        return float(np.mean(values))

    summary = {
        "rot_err_total_deg": mean("rot_err_total_deg"),
        "rot_err_tilt_deg": mean("rot_err_tilt_deg"),
        "rot_err_yaw_deg": mean("rot_err_yaw_deg"),
        "rot_err_total_deg_rest": mean("rot_err_total_deg_rest"),
        "rot_err_tilt_deg_rest": mean("rot_err_tilt_deg_rest"),
        "rot_err_yaw_deg_rest": mean("rot_err_yaw_deg_rest"),
        "rest_fraction": mean("rest_fraction"),
        "clean_lowest_vertex_to_plane_cm": mean("clean_lowest_vertex_to_plane_cm"),
        "clean_lowest_vertex_to_plane_cm_rest": mean("clean_lowest_vertex_to_plane_cm_rest"),
        "tier2_lowest_vertex_to_plane_cm_rest": mean("tier2_lowest_vertex_to_plane_cm_rest"),
    }
    print(json.dumps(summary, indent=2))
    print("\nper episode: total / tilt / yaw deg, rest% , clean lowest vertex to plane cm, size cm")
    for episode, value in sorted(episodes.items()):
        low = value["clean_lowest_vertex_to_plane_cm_rest"]
        print(
            f"  e{episode:02d} {value['object'][:22]:22s} "
            f"{value['rot_err_total_deg']:5.1f} /{value['rot_err_tilt_deg']:5.1f} /{value['rot_err_yaw_deg']:5.1f}  "
            f"rest={value['rest_fraction'] * 100:4.0f}%  "
            f"low={'n/a' if low is None else f'{low:6.2f}'}  size={value['mesh_extent_cm']:5.0f}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "what": "Tier 2 orientation error split into support-recoverable tilt and unrecoverable yaw",
                    "rest_speed_m_per_s": REST_SPEED_M_PER_S,
                    "summary": summary,
                    "per_episode": episodes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
