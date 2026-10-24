"""The deployable support-contact cleaner for the Tier 2 object trajectory.

`tilt_yaw` measures what a support-contact cleaner *could* buy by constructing the answer from
the clean reference. That is an oracle and a submitted pipeline may not use it. This is the
deployable version: it sees the Tier 2 object pose, the object mesh, the released ground plane
and the released support primitives, and never the clean trajectory.

The argument it rests on: an object at rest on a flat support has two of its three rotational
degrees of freedom pinned, because the supporting face must lie flush against the surface. The
third, a rotation about the surface normal, is free. So from rest frames alone one can recover
the **tilt** and nothing else -- which is exactly the component worth recovering, since the
oracle shows tilt restores 3.8x more contact per degree than yaw.

Per episode:

1. **Rest detection** on a smoothed trajectory. Tier 2's own position jitter differentiated at
   30 fps is ~0.2 m/s, an order of magnitude above the rest threshold, so without smoothing a
   resting object looks like a moving one (1.2 % of frames detected as rest instead of 38 %).
2. **Support height per rest segment**, the median lowest-vertex height above the released floor
   plane. Fitted per segment this recovers real furniture heights from noisy data alone:
   `traffic_cone_ground_table` reads -7 cm then 74.8 cm.
3. **A segment is corrected only where a support plane is known** -- see `v2d.supports` for the
   candidate heights and the asymmetric tolerances. Motionless is not supported: a watering can
   held steady in mid-air while pouring produces seven "rest" segments at 13-14 cm and ships no
   support primitive, so it is refused.
4. **Support facet per segment**: the hull facet whose outward normal best matches the object's
   downward direction, chosen once per segment by summed agreement so it cannot jitter.
5. **Tilt correction**: the minimal rotation taking that facet's normal onto the support normal.
   Being minimal, its axis lies in the support plane, so yaw is untouched by construction.
6. **Propagation**: 44 % of Tier 2's orientation error is a fixed per-clip offset, so the
   correction is informative away from rest too. Off-rest frames take a slerp between the
   bracketing segments' corrections, held flat outside the first and last.
7. **Optional vertical seating**, off by default. Measured and rejected: it interpolates a target
   height across unsupported frames and drags the object far off (AUC 0.2786, MPPE 18.94 cm).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..config import FPS
from ..dataset import (
    Episode,
    episodes,
    hold_last_visible,
    object_hull,
    object_pose,
    oriented_ground_plane,
    tier_root,
)
from ..geometry import minimal_rotation, rest_segments
from ..supports import ABOVE_TOLERANCE_CM, BELOW_TOLERANCE_CM, candidate_heights, match_support


@dataclass
class Options:
    """Everything that changes the output, so a record of it explains the output."""

    fps: float = FPS
    rest_speed: float = 0.02
    min_rest_frames: int = 5
    smooth_frames: int = 9
    max_tilt_deg: float = 45.0
    support_tolerance_cm: float = ABOVE_TOLERANCE_CM
    support_penetration_cm: float = BELOW_TOLERANCE_CM
    support_primitives: bool = False
    any_support_height: bool = False
    seat_vertical: bool = False

    def as_summary(self) -> dict[str, Any]:
        return {
            "rest_speed_m_per_s": self.rest_speed,
            "smooth_frames": self.smooth_frames,
            "support_tolerance_cm": self.support_tolerance_cm,
            "support_penetration_cm": self.support_penetration_cm,
            "support_primitives": self.support_primitives,
            "any_support_height": self.any_support_height,
            "seat_vertical": self.seat_vertical,
        }


@dataclass
class EpisodeRecord:
    """Per-episode diagnostics, written verbatim into the results JSON."""

    frames: int
    rest_segments: int
    rest_frames: int
    status: str = ""
    support_heights_cm: list[float] = field(default_factory=list)
    matched_support_cm: list[float] = field(default_factory=list)
    support_facets: list[int] = field(default_factory=list)
    segments_rejected_off_floor_cm: list[float] = field(default_factory=list)
    support_candidates_cm: list[float] = field(default_factory=list)
    corrected_frames: int = 0
    tilt_applied_deg_mean: float = 0.0
    tilt_applied_deg_max: float = 0.0
    vertical_shift_cm: list[float] = field(default_factory=list)
    object: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {
            "frames": self.frames,
            "rest_segments": self.rest_segments,
            "rest_frames": self.rest_frames,
            "status": self.status,
            "support_heights_cm": self.support_heights_cm,
            "segments_rejected_off_floor_cm": self.segments_rejected_off_floor_cm,
            "support_candidates_cm": self.support_candidates_cm,
            "object": self.object,
        }
        if self.matched_support_cm:
            out["matched_support_cm"] = self.matched_support_cm
        if self.support_facets:
            out["support_facets"] = self.support_facets
        if self.corrected_frames:
            out["corrected_frames"] = self.corrected_frames
            out["tilt_applied_deg_mean"] = self.tilt_applied_deg_mean
            out["tilt_applied_deg_max"] = self.tilt_applied_deg_max
        if self.vertical_shift_cm:
            out["vertical_shift_cm"] = self.vertical_shift_cm
        return out


def clean_episode(
    pose: np.ndarray,
    vertices: np.ndarray,
    hull_normals: np.ndarray,
    normal: np.ndarray,
    offset: float,
    candidates: list[float],
    options: Options,
) -> tuple[np.ndarray, EpisodeRecord]:
    """The cleaned ``(T,7)`` trajectory and this episode's diagnostics."""
    frames = len(pose)
    rotation = Rotation.from_quat(pose[:, 3:])
    matrices = rotation.as_matrix()

    segments = rest_segments(
        pose[:, :3],
        options.fps,
        options.rest_speed,
        options.min_rest_frames,
        options.smooth_frames,
    )
    record = EpisodeRecord(
        frames=frames,
        rest_segments=len(segments),
        rest_frames=sum(stop - start for start, stop in segments),
    )
    corrections: dict[int, Rotation] = {}
    applied: list[float] = []

    for start, stop in segments:
        span = slice(start, stop)
        # The object-frame direction pointing down in the world, summed over the segment.
        down_object = -np.einsum("tji,j->ti", matrices[span], normal)
        facet = int(np.argmax(hull_normals @ down_object.sum(axis=0)))
        world = np.einsum("tij,vj->tvi", matrices[span], vertices) + pose[span, None, :3]
        height = float(np.median((world @ normal + offset).min(axis=1)))
        record.support_heights_cm.append(round(height * 100.0, 2))

        if options.any_support_height:
            matched: float | None = 0.0
        else:
            matched = match_support(
                height,
                candidates,
                options.support_tolerance_cm / 100.0,
                options.support_penetration_cm / 100.0,
            )
            if matched is None:
                record.segments_rejected_off_floor_cm.append(round(height * 100.0, 2))
                continue
        record.matched_support_cm.append(round(matched * 100.0, 2))

        for index in range(start, stop):
            correction = minimal_rotation(matrices[index] @ hull_normals[facet], -normal)
            angle = float(np.degrees(correction.magnitude()))
            # A correction past the cap is not a tilt estimate, it is a facet mix-up.
            if angle <= options.max_tilt_deg:
                corrections[index] = correction
                applied.append(angle)
        record.support_facets.append(facet)

    if not corrections:
        record.status = (
            "no floor-level rest segments; left unchanged"
            if segments
            else "no rest frames; left unchanged"
        )
        return pose.copy(), record

    keys = np.array(sorted(corrections))
    key_rotations = Rotation.concatenate([corrections[int(key)] for key in keys])
    if len(keys) == 1:
        propagated = Rotation.concatenate([key_rotations[0]] * frames)
    else:
        propagated = Slerp(keys, key_rotations)(np.clip(np.arange(frames), keys[0], keys[-1]))

    cleaned_rotation = propagated * rotation
    cleaned = pose.copy()
    cleaned[:, 3:] = cleaned_rotation.as_quat()

    if options.seat_vertical:
        world = (
            np.einsum("tij,vj->tvi", cleaned_rotation.as_matrix(), vertices) + cleaned[:, None, :3]
        )
        lowest = (world @ normal + offset).min(axis=1)
        target = np.full(frames, np.nan)
        for (start, stop), height_cm in zip(segments, record.support_heights_cm, strict=False):
            seat = (
                0.0
                if options.any_support_height
                else match_support(
                    height_cm / 100.0,
                    candidates,
                    options.support_tolerance_cm / 100.0,
                    options.support_penetration_cm / 100.0,
                )
            )
            if seat is not None:
                target[start:stop] = seat
        if np.isfinite(target).any():
            index = np.arange(frames)
            finite = np.isfinite(target)
            shift = np.interp(index, index[finite], target[finite]) - lowest
            cleaned[:, :3] += shift[:, None] * normal
            record.vertical_shift_cm = [
                round(float(np.abs(shift).mean() * 100), 3),
                round(float(np.abs(shift).max() * 100), 3),
            ]

    record.corrected_frames = len(corrections)
    record.tilt_applied_deg_mean = round(float(np.mean(applied)), 3)
    record.tilt_applied_deg_max = round(float(np.max(applied)), 3)
    record.status = "cleaned"
    return cleaned, record


def write_npz(path: Path, episode: int, cleaned: np.ndarray) -> None:
    """The npz contract every consumer of a cleaned reference reads."""
    np.savez(
        path,
        episode_index=episode,
        coordinate_frame="released_track2_world",
        frame_index=np.arange(len(cleaned)),
        pose_xyzw=cleaned.reshape(len(cleaned), 1, 7),
    )


def run(
    dataset_root: Path,
    npz_out: Path,
    options: Options,
    only: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Clean every episode's object trajectory and write the npz the pipeline consumes."""
    npz_out.mkdir(parents=True, exist_ok=True)
    hulls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records: dict[int, dict[str, Any]] = {}

    for episode in episodes(dataset_root):
        if only is not None and episode.index not in only:
            continue
        noisy, visible = object_pose(dataset_root, "tier2", episode.index)
        noisy, _ = hold_last_visible(noisy, visible)
        normal, offset = oriented_ground_plane(dataset_root, episode, noisy)

        if episode.object_name not in hulls:
            hulls[episode.object_name] = object_hull(dataset_root, episode)
        vertices, hull_normals = hulls[episode.object_name]

        candidates = candidate_heights(
            tier_root(dataset_root, "tier1") / episode.support_surface
            if options.support_primitives
            else None
        )
        cleaned, record = clean_episode(
            noisy, vertices, hull_normals, normal, offset, candidates, options
        )
        record.object = episode.object_name
        record.support_candidates_cm = [round(value * 100.0, 2) for value in candidates]
        records[episode.index] = record.as_dict()
        write_npz(npz_out / f"episode_{episode.index:06d}.npz", episode.index, cleaned)

        if verbose:
            print(
                f"e{episode.index:02d} {episode.object_name[:22]:22s} {record.status:22s} "
                f"rest={record.rest_frames:4d}/{record.frames:4d} "
                f"segments={record.rest_segments}  "
                f"tilt={record.tilt_applied_deg_mean:5.2f} deg  "
                f"support_h={record.support_heights_cm}"
            )

    cleaned_count = sum(1 for value in records.values() if value["status"] == "cleaned")
    summary = {
        "episodes": len(records),
        "cleaned": cleaned_count,
        "unchanged": len(records) - cleaned_count,
        **options.as_summary(),
        "mean_rest_fraction": round(
            float(np.mean([v["rest_frames"] / v["frames"] for v in records.values()])), 4
        ),
        "mean_tilt_applied_deg": round(
            float(
                np.mean(
                    [
                        v["tilt_applied_deg_mean"]
                        for v in records.values()
                        if "tilt_applied_deg_mean" in v
                    ]
                )
            ),
            3,
        ),
    }
    return {
        "what": "deployable support-contact tilt cleaner; uses no clean reference",
        "summary": summary,
        "per_episode": records,
    }


def save(record: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


__all__ = [
    "Episode",
    "EpisodeRecord",
    "Options",
    "clean_episode",
    "run",
    "save",
    "write_npz",
]
