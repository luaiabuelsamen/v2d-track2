"""Per-hand contact ROLES for arm CR, from the reference the arm trains on.

The reward term arm CR adds is gated per hand on a role: *contact*, meaning this hand is supposed
to be holding the object on this frame, or *free*, meaning it is not and should not be rewarded for
where it is. The obvious source is the reference's own `hand_contact_active`, and that is the stub
the patch falls back to, but measured against the references it is noisy in two ways that both
hurt:

- **It drops out mid-grasp.** A hand carrying the object for 300 frames has the flag flicker off
  for a handful of frames at a time. A role that flickers makes the reward flicker, and the term
  the policy is being taught to maximise stops meaning "hold it" and starts meaning "hold it
  except on these frames". Gaps of `--fill-gap` frames or fewer (default 10, a third of a second)
  are filled.
- **It is set for hands that are nowhere near the object.** On e04 the right hand is flagged as in
  contact while its palm sits 10-17 cm from the mug's surface -- `docs/FAILURE_MOMENTS.md` has the
  numbers. Rewarding that hand for closing a gap it is not supposed to close teaches the policy to
  reach for the object with a hand the reference keeps away from it. A flagged frame only counts
  as a contact role if the palm is within `--max-distance` of the object's surface (default 8 cm).

Both filters are applied here, offline, on the reference -- not in the warp kernel -- so what the
policy was taught is a file that can be read, diffed and scored, and so the kernel keeps one
cheap array lookup per hand. The surface distance is the kit's own AUC vertex cloud, the same
"surface" the score is computed over, rather than the bounding sphere the kernel's reward uses.

Writes one `eXX.npz` per clip holding `role` of shape (frames, 2), side order `("left", "right")`,
which is exactly what `brev/patch_contact_role.py` reads from `$V2D_CONTACT_ROLES`.

    v2d contact-roles --clips e01,e04 --motion-root $S/t2_clean/tier2 \
      --dataset-root $S/data --kit-root $KIT --out results/contact_roles
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..analyze.deep_dive import surface_distance
from ..analyze.failure_moments import motion_for, palm_positions, reference_state
from ..dataset import episodes as dataset_episodes

TIER1 = "tier_1_multiview_caption"
SIDES = ("left", "right")
DEFAULT_FILL_GAP = 10
DEFAULT_MAX_DISTANCE_CM = 8.0


def reference_object_pose(parquet: Path) -> np.ndarray:
    """`(T, 7)` object pose, xyz + xyzw quaternion, from the reference the arm trains on.

    The tree stores the rotation as `object_root_axis_angle` -- a rotation vector whose magnitude
    is the angle -- while the kit's vertex machinery wants xyzw, so it is converted here rather
    than borrowing the rollout's `reference_*` columns: those carry whichever reference that
    policy trained on, and arm CR is defined against bridge_v1 for every clip.
    """
    row = pq.read_table(parquet, columns=["object_root_position", "object_root_axis_angle"])
    position = np.asarray(row["object_root_position"][0].as_py(), dtype=np.float64)
    rotation = np.asarray(row["object_root_axis_angle"][0].as_py(), dtype=np.float64)
    angle = np.linalg.norm(rotation, axis=-1)
    axis = np.divide(
        rotation, angle[:, None], out=np.zeros_like(rotation), where=angle[:, None] > 0.0
    )
    half = 0.5 * angle
    quaternion = np.concatenate([axis * np.sin(half)[:, None], np.cos(half)[:, None]], axis=1)
    return np.concatenate([position, quaternion], axis=1)


def fill_gaps(flag: np.ndarray, max_gap: int) -> np.ndarray:
    """`flag` with every run of zeros of length <= max_gap *between two ones* set to one.

    Leading and trailing zeros are left alone: a gap is only a gap if the hand is holding the
    object on both sides of it. Reaching for the object and letting go of it are not dropouts.
    """
    out = np.asarray(flag, dtype=bool).copy()
    if max_gap <= 0 or not out.any():
        return out
    ones = np.flatnonzero(out)
    for start, stop in itertools.pairwise(ones):
        if 1 < stop - start <= max_gap + 1:
            out[start:stop] = True
    return out


def roles_for_clip(
    state: dict[str, np.ndarray],
    object_pose: np.ndarray,
    vertices: np.ndarray,
    *,
    fill_gap: int,
    max_distance_cm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """`(frames, 2)` roles plus what each filter changed, for the record."""
    contact = np.asarray(state["contact_active"], dtype=np.float64)
    palms = palm_positions(state)
    frames = min(len(contact), len(palms), len(object_pose))
    contact = contact[:frames]
    palms = palms[:frames]
    pose = object_pose[:frames]
    if contact.shape[1] < len(SIDES):
        contact = np.pad(contact, ((0, 0), (0, len(SIDES) - contact.shape[1])))

    roles = np.zeros((frames, len(SIDES)), dtype=np.float32)
    per_hand = []
    for hand, side in enumerate(SIDES):
        raw = contact[:, hand] > 0.5
        filled = fill_gaps(raw, fill_gap)
        distance_cm = surface_distance(palms[:, hand], pose, vertices) * 100.0
        near = distance_cm <= max_distance_cm
        kept = filled & near
        roles[:, hand] = kept.astype(np.float32)
        held = distance_cm[filled]
        per_hand.append(
            {
                "side": side,
                "flagged_frames": int(raw.sum()),
                "after_gap_fill": int(filled.sum()),
                "gap_frames_filled": int(filled.sum() - raw.sum()),
                "contact_role_frames": int(kept.sum()),
                "frames_dropped_as_too_far": int((filled & ~near).sum()),
                "palm_to_surface_cm_while_flagged": {
                    "min": round(float(held.min()), 2) if held.size else None,
                    "median": round(float(np.median(held)), 2) if held.size else None,
                    "max": round(float(held.max()), 2) if held.size else None,
                },
            }
        )
    record = {
        "frames": int(frames),
        "hands": per_hand,
        "frames_with_any_contact_role": int((roles > 0.5).any(axis=1).sum()),
        "frames_with_both_hands": int((roles > 0.5).all(axis=1).sum()),
    }
    return roles, record


def write_roles(path: Path, episode: int, roles: np.ndarray, provenance: dict[str, Any]) -> None:
    """The npz the patch reads: `role` (frames, 2), plus enough provenance to audit it."""
    np.savez(
        path,
        role=roles.astype(np.float32),
        episode_index=np.int64(episode),
        sides=np.asarray(SIDES),
        provenance=np.asarray(json.dumps(provenance)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, help="eXX,eYY,... or 'all'")
    parser.add_argument(
        "--motion-root", type=Path, required=True, help="the reference tree the arm trains on"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True, help="for the kit's vertex sampler")
    parser.add_argument("--out", type=Path, required=True, help="directory for the eXX.npz files")
    parser.add_argument("--json", dest="record_out", type=Path, help="write the summary here")
    parser.add_argument("--fill-gap", type=int, default=DEFAULT_FILL_GAP)
    parser.add_argument("--max-distance-cm", type=float, default=DEFAULT_MAX_DISTANCE_CM)
    args = parser.parse_args(argv)

    if args.fill_gap < 0:
        raise SystemExit(f"--fill-gap must be >= 0, got {args.fill_gap}")
    if not args.max_distance_cm > 0:
        raise SystemExit(f"--max-distance-cm must be positive, got {args.max_distance_cm}")

    metadata = {f"e{e.index:02d}": e for e in dataset_episodes(args.dataset_root)}
    wanted = (
        sorted(metadata)
        if args.clips.strip() == "all"
        else [tag.strip() for tag in args.clips.split(",") if tag.strip()]
    )
    unknown = [tag for tag in wanted if tag not in metadata]
    if unknown:
        raise SystemExit(f"unknown clips: {unknown}")

    if str(args.kit_root) not in sys.path:
        sys.path.insert(0, str(args.kit_root))
    from v2dlb.vertices import build_blob

    _blob, vertex_cloud = build_blob(
        args.dataset_root / "track_2" / TIER1 / "mesh",
        sorted({metadata[tag].object_name for tag in wanted}),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    clips = {}
    for tag in wanted:
        episode = metadata[tag]
        parquet = motion_for(args.motion_root, episode.sequence_id)
        state = reference_state(parquet)
        pose = reference_object_pose(parquet)
        roles, record = roles_for_clip(
            state,
            pose,
            np.asarray(vertex_cloud[episode.object_name], dtype=np.float64),
            fill_gap=args.fill_gap,
            max_distance_cm=args.max_distance_cm,
        )
        record["object"] = episode.object_name
        record["reference"] = str(args.motion_root)
        provenance = {
            "tag": tag,
            "reference": str(args.motion_root),
            "fill_gap_frames": args.fill_gap,
            "max_palm_to_surface_cm": args.max_distance_cm,
            "surface": "the kit's AUC vertex cloud",
            "sides": list(SIDES),
        }
        write_roles(args.out / f"{tag}.npz", episode.index, roles, provenance)
        clips[tag] = record
        hands = " ".join(
            f"{h['side']}={h['contact_role_frames']}(+{h['gap_frames_filled']}"
            f"/-{h['frames_dropped_as_too_far']})"
            for h in record["hands"]
        )
        print(f"{tag} {record['frames']} frames | contact-role frames {hands}")

    summary = {
        "what": "per-hand contact roles for arm CR, from the reference the arm trains on",
        "reference": str(args.motion_root),
        "fill_gap_frames": args.fill_gap,
        "max_palm_to_surface_cm": args.max_distance_cm,
        "out": str(args.out),
        "clips": clips,
        "totals": {
            "gap_frames_filled": sum(
                h["gap_frames_filled"] for c in clips.values() for h in c["hands"]
            ),
            "frames_dropped_as_too_far": sum(
                h["frames_dropped_as_too_far"] for c in clips.values() for h in c["hands"]
            ),
            "contact_role_frames": sum(
                h["contact_role_frames"] for c in clips.values() for h in c["hands"]
            ),
        },
    }
    if args.record_out:
        args.record_out.write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"{len(clips)} clips | filled {summary['totals']['gap_frames_filled']} gap frames | "
        f"dropped {summary['totals']['frames_dropped_as_too_far']} flagged frames as too far"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
