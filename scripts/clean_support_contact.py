"""A deployable support-contact cleaner for the Tier 2 object trajectory.

`tilt_yaw_split.py` measures what a support-contact cleaner *could* buy by constructing the
answer from the clean reference. That is an oracle and a submitted pipeline may not use it.
This script is the deployable version: it sees only the Tier 2 object pose, the object mesh and
the released ground-plane normal, and never the clean trajectory.

The argument it rests on: an object at rest on a flat support has two of its three rotational
degrees of freedom pinned, because the supporting face must lie flush against the surface. The
third, a rotation about the surface normal, is free. So from rest frames alone one can recover
the *tilt* and nothing else -- which is exactly the component the oracle showed is worth
recovering (it restores 3.8x more contact per degree than yaw, see docs/EXP_CLEANUP_CONTACT.md).

Per episode:

1. **Rest detection.** Object-origin speed below `--rest-speed` m/s, in contiguous runs of at
   least `--min-rest-frames`, computed from the noisy trajectory so it stays deployable. The
   speed is measured on a `--smooth-frames` moving average of the position, because Tier 2's own
   position noise otherwise masquerades as motion: 0.73 cm of within-clip jitter differentiated
   at 30 fps is ~0.21 m/s, an order of magnitude above the rest threshold, which leaves a
   resting object looking like a moving one.
2. **Support height per rest segment**, as the median lowest-vertex height above the released
   floor plane over that segment's frames. Fitting this per segment recovers the real furniture
   heights from noisy data alone: `traffic_cone_ground_table` reads -7 cm then 74.8 cm,
   `woven_basket_ground_desk` reads ~0 then 83 cm, matching the task names.
3. **Only floor-level segments are used**: a segment qualifies when its lowest vertex sits no
   more than `--floor-tolerance-cm` (default 3) *above* the plane. Penetration is accepted
   without bound, because a tilt error is precisely what digs a corner into the floor -- a 20°
   error on an 86 cm box buries one by ~15 cm -- so gating on absolute distance would discard
   exactly the frames worth correcting. Motionless is not
   the same as supported: a watering can held steady in mid-air while pouring produces seven
   "rest" segments at 13-14 cm, and seating it on a plane tangent to itself would invent a
   constraint that does not exist. At floor level the support plane is known and was validated
   to +-1.5 cm, so the correction is sound exactly where it is applied. Extending it to tables
   needs the released `support_surface/` primitives, which ship in a different frame with no
   published transform -- deliberately out of scope here.
4. **Support facet per rest segment.** The facet of the mesh's convex hull whose outward normal
   best matches the object's current downward direction, chosen once per segment by summed
   agreement so it cannot jitter frame to frame.
5. **Tilt correction.** The minimal world rotation taking that facet's normal onto the support
   normal. Being the minimal rotation between two vectors, its axis is perpendicular to both and
   so lies in the support plane: it is a pure tilt correction and leaves yaw untouched by
   construction.
6. **Propagation.** 44 % of Tier 2's orientation error is a fixed per-clip offset, so the
   correction is informative away from rest too. Off-rest frames take a slerp between the
   bracketing segments' corrections, held flat outside the first and last segment.
7. **Optional vertical seating** (`--seat-vertical`): shift the object along the support normal
   so the supporting facet touches rather than penetrates. Off by default, so the default output
   changes orientation only and stays comparable to the oracle arms.

Output matches `tilt_yaw_split.py`'s npz contract, so `score_track2.py --pred npz:DIR` and
`retarget_all.py --object-npz DIR` both consume it unchanged.

Usage:
  python scripts/clean_support_contact.py --dataset-root DATA --npz-out DIR \
      [--seat-vertical] [-o results/clean_support_contact.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

TIER1, TIER2 = "tier_1_multiview_caption", "tier_2_synthetic_noise"
MAX_HULL_FACETS = 400


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Released parquet -> (T,7) xyz+xyzw and the visibility mask."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(path, columns=["observation.object.pose", "observation.object.visible"])
    pose = np.asarray(table["observation.object.pose"].to_pylist(), dtype=np.float64)
    visible = np.asarray(table["observation.object.visible"].to_pylist(), dtype=bool)
    # released order is xyz + wxyz
    return np.concatenate([pose[:, :3], pose[:, [4, 5, 6, 3]]], axis=1), visible


def hold_last_visible(pose: np.ndarray, visible: np.ndarray) -> np.ndarray:
    if visible.all():
        return pose
    index = np.arange(len(visible))
    index[~visible] = -1
    index = np.maximum.accumulate(index)
    if index[0] < 0:
        index[index < 0] = int(np.argmax(visible)) if visible.any() else 0
    return pose[index]


def smooth(position: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding, so differentiating does not amplify the noise."""
    if window <= 1:
        return position
    pad = window // 2
    padded = np.pad(position, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid")[: len(position)]
                     for axis in range(position.shape[1])], axis=1)


def rest_segments(position: np.ndarray, fps: float, speed_limit: float, min_frames: int,
                  window: int = 1) -> list[tuple[int, int]]:
    """Contiguous [start, stop) runs where the object origin is slower than `speed_limit`."""
    speed = np.linalg.norm(np.gradient(smooth(position, window), axis=0), axis=-1) * fps
    rest = speed < speed_limit
    segments, start = [], None
    for index, value in enumerate(rest):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_frames:
                segments.append((start, index))
            start = None
    if start is not None and len(rest) - start >= min_frames:
        segments.append((start, len(rest)))
    return segments


def minimal_rotation(source: np.ndarray, target: np.ndarray) -> Rotation:
    """The minimal rotation taking unit vector `source` onto unit vector `target`.

    Its axis is source x target, perpendicular to the support normal, so applying it tilts the
    object without rotating it about that normal -- a pure tilt correction.
    """
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    axis = np.cross(source, target)
    norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if norm < 1e-12:
        if dot > 0:
            return Rotation.identity()
        # Antiparallel: any perpendicular axis gives a 180 deg turn. Pick a stable one.
        helper = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi)
    return Rotation.from_rotvec(axis / norm * np.arccos(dot))


def clean_episode(pose: np.ndarray, vertices: np.ndarray, hull_normals: np.ndarray,
                  normal: np.ndarray, offset: float, args) -> tuple[np.ndarray, dict]:
    """Return the cleaned (T,7) trajectory and this episode's diagnostics."""
    frames = len(pose)
    rotation = Rotation.from_quat(pose[:, 3:])
    matrices = rotation.as_matrix()

    segments = rest_segments(pose[:, :3], args.fps, args.rest_speed, args.min_rest_frames,
                             args.smooth_frames)
    corrections: dict[int, Rotation] = {}
    diagnostics = {"rest_segments": len(segments), "rest_frames": sum(b - a for a, b in segments),
                   "frames": frames, "support_heights_cm": [], "tilt_applied_deg": []}

    rejected = []
    for start, stop in segments:
        span = slice(start, stop)
        # The object-frame direction that points down in the world, averaged over the segment.
        down_object = -np.einsum("tji,j->ti", matrices[span], normal)
        # One support facet for the whole segment, by summed agreement, so it cannot jitter.
        facet = int(np.argmax(hull_normals @ down_object.sum(axis=0)))
        # Support height: median lowest-vertex distance to the floor plane over the segment.
        world = np.einsum("tij,vj->tvi", matrices[span], vertices) + pose[span, None, :3]
        lowest = (world @ normal + offset).min(axis=1)
        height = float(np.median(lowest))
        diagnostics["support_heights_cm"].append(round(height * 100.0, 2))
        # Only the floor is a support plane we actually know. See the module docstring.
        if not args.any_support_height and height > args.floor_tolerance_cm / 100.0:
            rejected.append(round(height * 100.0, 2))
            continue

        for index in range(start, stop):
            current = matrices[index] @ hull_normals[facet]
            correction = minimal_rotation(current, -normal)
            angle = float(np.degrees(correction.magnitude()))
            # A correction larger than the cap is not a tilt estimate, it is a facet mix-up.
            if angle <= args.max_tilt_deg:
                corrections[index] = correction
                diagnostics["tilt_applied_deg"].append(angle)
        diagnostics.setdefault("support_facets", []).append(facet)

    diagnostics["segments_rejected_off_floor_cm"] = rejected
    if not corrections:
        diagnostics["status"] = ("no floor-level rest segments; left unchanged" if segments
                                 else "no rest frames; left unchanged")
        return pose.copy(), diagnostics

    # Propagate: slerp between the frames that have a correction, flat outside their range.
    keys = np.array(sorted(corrections))
    key_rotations = Rotation.concatenate([corrections[int(k)] for k in keys])
    if len(keys) == 1:
        applied = Rotation.concatenate([key_rotations[0]] * frames)
    else:
        slerp = Slerp(keys, key_rotations)
        clamped = np.clip(np.arange(frames), keys[0], keys[-1])
        applied = slerp(clamped)

    cleaned_rotation = applied * rotation
    cleaned = pose.copy()
    cleaned[:, 3:] = cleaned_rotation.as_quat()

    if args.seat_vertical:
        cleaned_matrices = cleaned_rotation.as_matrix()
        world = np.einsum("tij,vj->tvi", cleaned_matrices, vertices) + cleaned[:, None, :3]
        lowest = (world @ normal + offset).min(axis=1)
        target = np.full(frames, np.nan)
        for (start, stop), height in zip(segments, diagnostics["support_heights_cm"]):
            if args.any_support_height or height <= args.floor_tolerance_cm:
                target[start:stop] = height / 100.0
        if np.isfinite(target).any():
            index = np.arange(frames)
            finite = np.isfinite(target)
            target = np.interp(index, index[finite], target[finite])
            shift = target - lowest
            cleaned[:, :3] += shift[:, None] * normal
            diagnostics["vertical_shift_cm"] = [round(float(np.abs(shift).mean() * 100), 3),
                                                round(float(np.abs(shift).max() * 100), 3)]

    diagnostics["tilt_applied_deg_mean"] = round(float(np.mean(diagnostics["tilt_applied_deg"])), 3)
    diagnostics["tilt_applied_deg_max"] = round(float(np.max(diagnostics["tilt_applied_deg"])), 3)
    diagnostics["corrected_frames"] = len(corrections)
    del diagnostics["tilt_applied_deg"]
    diagnostics["status"] = "cleaned"
    return cleaned, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--npz-out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rest-speed", type=float, default=0.02, help="m/s below which a frame is at rest")
    parser.add_argument("--min-rest-frames", type=int, default=5)
    parser.add_argument("--smooth-frames", type=int, default=9,
                        help="moving-average window for the speed estimate; 1 disables it")
    parser.add_argument("--max-tilt-deg", type=float, default=45.0,
                        help="reject a per-frame correction above this as a facet mix-up")
    parser.add_argument("--floor-tolerance-cm", type=float, default=3.0,
                        help="how far ABOVE the floor a resting object's lowest vertex may sit and "
                             "still count as floor-supported; penetration is unbounded, because a "
                             "tilt error digs a corner in and that is the error being corrected")
    parser.add_argument("--any-support-height", action="store_true",
                        help="also correct rest segments away from the floor; see the docstring")
    parser.add_argument("--seat-vertical", action="store_true",
                        help="also shift along the support normal so the facet touches")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    root = args.dataset_root / "track_2" / TIER1
    metadata = [json.loads(l) for l in (root / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()]
    metadata.sort(key=lambda r: r["episode_index"])
    args.npz_out.mkdir(parents=True, exist_ok=True)

    hulls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    episodes = {}
    for row in metadata:
        episode = int(row["episode_index"])
        if args.episodes is not None and episode not in args.episodes:
            continue
        name = row["object"]
        noisy, visible = load_pose(
            args.dataset_root / "track_2" / TIER2 / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        )
        noisy = hold_last_visible(noisy, visible)

        plane = np.asarray(json.loads((root / row["ground_plane"]).read_text())["plane"], dtype=np.float64)
        normal = plane[:3] / np.linalg.norm(plane[:3])
        offset = plane[3] / np.linalg.norm(plane[:3])
        # Orient the plane so the object sits on the +side. Uses the noisy trajectory only.
        if np.mean(noisy[:, :3] @ normal + offset) < 0:
            normal, offset = -normal, -offset

        if name not in hulls:
            mesh = trimesh.load(str(root / row["mesh"]), force="mesh", process=False)
            hull = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)).convex_hull
            normals = np.asarray(hull.face_normals, dtype=np.float64)
            areas = np.asarray(hull.area_faces, dtype=np.float64)
            if len(normals) > MAX_HULL_FACETS:  # keep the largest facets; tiny ones cannot support
                keep = np.argsort(areas)[::-1][:MAX_HULL_FACETS]
                normals = normals[keep]
            hulls[name] = (np.asarray(hull.vertices, dtype=np.float64), normals)
        vertices, hull_normals = hulls[name]

        cleaned, diagnostics = clean_episode(noisy, vertices, hull_normals, normal, offset, args)
        diagnostics["object"] = name
        episodes[episode] = diagnostics

        np.savez(
            args.npz_out / f"episode_{episode:06d}.npz",
            episode_index=episode,
            coordinate_frame="released_track2_world",
            frame_index=np.arange(len(cleaned)),
            pose_xyzw=cleaned.reshape(len(cleaned), 1, 7),
        )
        print(f"e{episode:02d} {name[:22]:22s} {diagnostics['status']:22s} "
              f"rest={diagnostics['rest_frames']:4d}/{diagnostics['frames']:4d} "
              f"segments={diagnostics['rest_segments']}  "
              f"tilt={diagnostics.get('tilt_applied_deg_mean', 0):5.2f} deg  "
              f"support_h={diagnostics['support_heights_cm']}")

    summary = {
        "episodes": len(episodes),
        "cleaned": sum(1 for v in episodes.values() if v["status"] == "cleaned"),
        "unchanged": sum(1 for v in episodes.values() if v["status"] != "cleaned"),
        "seat_vertical": bool(args.seat_vertical),
        "rest_speed_m_per_s": args.rest_speed,
        "smooth_frames": args.smooth_frames,
        "floor_tolerance_cm": args.floor_tolerance_cm,
        "any_support_height": bool(args.any_support_height),
        "mean_rest_fraction": round(float(np.mean([v["rest_frames"] / v["frames"] for v in episodes.values()])), 4),
        "mean_tilt_applied_deg": round(float(np.mean(
            [v["tilt_applied_deg_mean"] for v in episodes.values() if "tilt_applied_deg_mean" in v])), 3),
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"what": "deployable support-contact tilt cleaner; uses no clean reference",
             "summary": summary, "per_episode": episodes}, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
