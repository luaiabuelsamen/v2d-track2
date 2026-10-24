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
3. **A segment is used only where a support plane is actually known.** Motionless is not the
   same as supported: a watering can held steady in mid-air while pouring produces seven "rest"
   segments at 13-14 cm, and seating it on a plane tangent to itself would invent a constraint
   that does not exist. So each segment must match a *candidate support height*, and candidates
   come from two places:

   - **The floor**, always, at height 0 -- validated to +-1.5 cm against the released plane.
   - **The released `support_surface/` primitives** with `--support-primitives`. These ship in a
     Z-up `g1_retarget_world` with no published transform to the pose frame, but a *height above
     the floor* is a scalar and survives any transform that preserves up and slides within the
     floor plane, so a primitive's top-face height is directly comparable to a measured one. Over
     the 96 non-floor rest segments the floor-only gate refuses, 73 land within 2 cm of a
     horizontal primitive top at a median gap of 0.61 cm (`scripts/probe_support_root.py`), which
     is also the evidence that the two frames share an up axis. Only *horizontal* primitives
     count -- thin in z, like a tabletop -- because a panel thin in x is something you lean
     against, not something you rest on, and `vacuum`'s only primitive is exactly that.

   Tolerances are asymmetric. A segment may sit at most `--support-tolerance-cm` (default 3)
   *above* its support, because an object at rest touches it. Penetration is accepted, because a
   tilt error is precisely what digs a corner through a surface -- a 20° error on an 86 cm box
   buries one by ~15 cm, `traffic_cone` sits 2.3 cm under its table and `woven_basket` 4-6 cm
   under its desk -- so gating on absolute distance would discard exactly the frames worth
   correcting. Below the floor penetration is unbounded, there being nothing further down to
   confuse it with; below a raised surface it is capped at `--support-penetration-cm` (default
   10), because the floor is down there. That cap is safe: the lowest table top in any episode
   sits 46.9 cm up, far outside the window.

   The correction itself is unchanged by which candidate matched, because a tabletop's normal is
   the floor's normal -- both frames describe one gravity-aligned room. The matched height enters
   only `--seat-vertical`.
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
      [--support-primitives] [--seat-vertical] [-o results/clean_support_contact.json]
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


def support_primitives(path: Path) -> list[float]:
    """Heights above the floor of every *horizontal* support primitive in a `_support.usda`.

    Horizontal means thin along z: a surface you rest on. A primitive thin along x or y is a
    panel you lean against, which pins no tilt, and `vacuum`'s `chair_back` is one of those.

    Duplicated from scripts/probe_support_root.py rather than imported: these tools run under
    `python -I`, which implies `-P`, so the script's own directory is not on sys.path and
    scripts/ cannot import from itself.
    """
    from pxr import Usd, UsdGeom  # noqa: PLC0415  (lazy: only needed with --support-primitives)

    axis_index = {"X": 0, "Y": 1, "Z": 2}
    stage = Usd.Stage.Open(str(path))
    heights = []
    for prim in stage.Traverse():
        kind = prim.GetTypeName()
        if kind not in ("Cylinder", "Cube", "Sphere"):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        centre = np.array(transform.ExtractTranslation(), dtype=np.float64)
        if kind == "Cylinder":
            cylinder = UsdGeom.Cylinder(prim)
            local = np.full(3, float(cylinder.GetRadiusAttr().Get()))
            local[axis_index[str(cylinder.GetAxisAttr().Get())]] = float(cylinder.GetHeightAttr().Get()) / 2.0
        elif kind == "Cube":
            local = np.full(3, float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2.0)
        else:
            local = np.full(3, float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        half = local * np.linalg.norm(np.array(transform, dtype=np.float64)[:3, :3], axis=1)
        if np.argmin(half) == 2:
            heights.append(float(centre[2] + half[2]))
    return sorted(heights)


def match_support(height: float, candidates: list[float], above: float, below: float) -> float | None:
    """The candidate support height a rest segment at `height` sits on, or None if none fits.

    Asymmetric by design; see the module docstring. Unbounded penetration below the floor,
    bounded below a raised surface.
    """
    best = min(candidates, key=lambda candidate: abs(height - candidate))
    gap = height - best
    if gap > above:
        return None
    if best > 0.0 and gap < -below:
        return None
    return best


def clean_episode(pose: np.ndarray, vertices: np.ndarray, hull_normals: np.ndarray,
                  normal: np.ndarray, offset: float, candidates: list[float],
                  args) -> tuple[np.ndarray, dict]:
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
        # Correct only where a support plane is actually known. See the module docstring.
        if args.any_support_height:
            matched = 0.0
        else:
            matched = match_support(height, candidates,
                                    args.support_tolerance_cm / 100.0,
                                    args.support_penetration_cm / 100.0)
            if matched is None:
                rejected.append(round(height * 100.0, 2))
                continue
        diagnostics.setdefault("matched_support_cm", []).append(round(matched * 100.0, 2))

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
            matched = (0.0 if args.any_support_height
                       else match_support(height / 100.0, candidates,
                                          args.support_tolerance_cm / 100.0,
                                          args.support_penetration_cm / 100.0))
            if matched is not None:
                target[start:stop] = matched
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
    parser.add_argument("--support-tolerance-cm", "--floor-tolerance-cm", type=float, default=3.0,
                        dest="support_tolerance_cm",
                        help="how far ABOVE a support a resting object's lowest vertex may sit and "
                             "still count as supported; penetration is allowed, because a tilt "
                             "error digs a corner in and that is the error being corrected")
    parser.add_argument("--support-penetration-cm", type=float, default=10.0,
                        help="how far BELOW a raised support a segment may sit and still match it; "
                             "unbounded below the floor, where nothing is further down")
    parser.add_argument("--support-primitives", action="store_true",
                        help="also accept rest on the released support_surface/ primitives, not "
                             "just on the floor; see the module docstring")
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

        candidates = [0.0]
        if args.support_primitives:
            candidates += support_primitives(root / row["support_surface"])
        cleaned, diagnostics = clean_episode(noisy, vertices, hull_normals, normal, offset,
                                             candidates, args)
        diagnostics["support_candidates_cm"] = [round(c * 100.0, 2) for c in candidates]
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
        "support_tolerance_cm": args.support_tolerance_cm,
        "support_penetration_cm": args.support_penetration_cm,
        "support_primitives": bool(args.support_primitives),
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
