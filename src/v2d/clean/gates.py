"""The three gates every cleaner must pass, in the order ~/projects/_ops/briefs says.

1. **Known-answer**: inject noise we generated ourselves into a clean trajectory and check the
   cleaner takes it back out. Without this, "AUC went up" could be luck, or could be a cleaner
   that happens to push every object toward the dataset's average pose.
2. **Do-no-harm**: run the cleaner on an already-clean trajectory and check it barely moves.
   This is the one that catches a cleaner which improves the mean by damaging good clips, and it
   is the cheapest possible guard against the failure the root cleaner actually had.
3. **Local score on real Tier 2** -- `v2d score`, reported elsewhere.

The injected noise is **ours**, built from a seed, and the clean Tier 1 trajectory is used only
to construct the test and to measure recovery. It is never an input or a target for anything
that runs on Tier 2, which is the standing rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..dataset import (
    episodes,
    hold_last_visible,
    object_hull,
    object_pose,
    oriented_ground_plane,
    tier_root,
)
from ..geometry import rest_segments
from ..supports import candidate_heights
from .support_contact import Options, clean_episode


@dataclass(frozen=True)
class Injection:
    """A synthetic tilt, of our own making, for the known-answer gate."""

    tilt_deg: float = 12.0
    seed: int = 0

    def apply(
        self, pose: np.ndarray, normal: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, Rotation]:
        """Tilt the whole trajectory by a fixed rotation about an axis in the support plane.

        A *tilt* specifically: the axis is perpendicular to the support normal, so this is the
        component the cleaner claims to recover, and a recovery figure means something. A
        rotation about the normal would be yaw, which no support-contact argument can recover and
        which the cleaner is built not to touch.
        """
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, normal))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(normal, helper)
        axis /= np.linalg.norm(axis)
        angle = np.radians(self.tilt_deg)
        # A random sign per clip, so a cleaner cannot pass by always tilting one way.
        signed = axis * angle * (1.0 if rng.random() < 0.5 else -1.0)
        injected = Rotation.from_rotvec(signed)
        out = pose.copy()
        out[:, 3:] = (injected * Rotation.from_quat(pose[:, 3:])).as_quat()
        return out, injected


def _mean_tilt_deg(
    pose_a: np.ndarray, pose_b: np.ndarray, normal: np.ndarray, where: np.ndarray
) -> float:
    """Mean angle between two orientation tracks over `where`, about axes in the support plane.

    Only the tilt component is compared, because that is the only component under test.
    """
    if not where.any():
        return float("nan")
    relative = Rotation.from_quat(pose_a[where, 3:]) * Rotation.from_quat(pose_b[where, 3:]).inv()
    rotvec = relative.as_rotvec()
    tilt = rotvec - np.outer(rotvec @ normal, normal)
    return float(np.degrees(np.linalg.norm(tilt, axis=1)).mean())


def run(
    dataset_root: Path,
    options: Options,
    injection: Injection = Injection(),
    only: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the known-answer and do-no-harm gates over every episode."""
    rng = np.random.default_rng(injection.seed)
    hulls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    for episode in episodes(dataset_root):
        if only is not None and episode.index not in only:
            continue
        clean, visible = object_pose(dataset_root, "tier1", episode.index)
        clean, _ = hold_last_visible(clean, visible)
        normal, offset = oriented_ground_plane(dataset_root, episode, clean)
        if episode.object_name not in hulls:
            hulls[episode.object_name] = object_hull(dataset_root, episode)
        vertices, hull_normals = hulls[episode.object_name]
        candidates = candidate_heights(
            tier_root(dataset_root, "tier1") / episode.support_surface
            if options.support_primitives
            else None
        )
        at_rest = np.zeros(len(clean), dtype=bool)
        for start, stop in rest_segments(
            clean[:, :3],
            options.fps,
            options.rest_speed,
            options.min_rest_frames,
            options.smooth_frames,
        ):
            at_rest[start:stop] = True

        # --- gate 1: inject our own tilt, then try to take it back out
        noisy, injected = injection.apply(clean, normal, rng)
        recovered, _ = clean_episode(
            noisy, vertices, hull_normals, normal, offset, candidates, options
        )
        before = _mean_tilt_deg(noisy, clean, normal, at_rest)
        after = _mean_tilt_deg(recovered, clean, normal, at_rest)

        # --- gate 2: the same cleaner on already-clean input must barely move it
        untouched, _ = clean_episode(
            clean, vertices, hull_normals, normal, offset, candidates, options
        )
        harm = _mean_tilt_deg(untouched, clean, normal, at_rest)
        drift_cm = float(np.linalg.norm(untouched[:, :3] - clean[:, :3], axis=1).max() * 100)

        rows.append(
            {
                "episode": episode.index,
                "object": episode.object_name,
                "rest_frames": int(at_rest.sum()),
                "frames": len(clean),
                "injected_deg": round(float(np.degrees(injected.magnitude())), 3),
                "tilt_before_deg": round(before, 3),
                "tilt_after_deg": round(after, 3),
                "recovered_fraction": (
                    None if not np.isfinite(before) or before == 0 else round(1 - after / before, 4)
                ),
                "do_no_harm_tilt_deg": round(harm, 3),
                "do_no_harm_drift_cm": round(drift_cm, 4),
            }
        )
        if verbose:
            row = rows[-1]
            print(
                f"e{episode.index:02d} {episode.object_name[:22]:22s} "
                f"injected {row['injected_deg']:5.2f} deg  "
                f"tilt {row['tilt_before_deg']:6.2f} -> {row['tilt_after_deg']:6.2f}  "
                f"recovered {row['recovered_fraction']}  "
                f"harm {row['do_no_harm_tilt_deg']:5.2f} deg / "
                f"{row['do_no_harm_drift_cm']:.3f} cm"
            )

    usable = [row for row in rows if row["recovered_fraction"] is not None]
    summary = {
        "episodes": len(rows),
        "episodes_with_rest": len(usable),
        "injected_tilt_deg": injection.tilt_deg,
        "seed": injection.seed,
        "mean_recovered_fraction": (
            round(float(np.mean([row["recovered_fraction"] for row in usable])), 4)
            if usable
            else None
        ),
        "worst_recovered_fraction": (
            round(min(row["recovered_fraction"] for row in usable), 4) if usable else None
        ),
        "mean_do_no_harm_tilt_deg": round(
            float(np.mean([row["do_no_harm_tilt_deg"] for row in rows])), 4
        ),
        "worst_do_no_harm_tilt_deg": round(max(row["do_no_harm_tilt_deg"] for row in rows), 4),
        "worst_do_no_harm_drift_cm": round(max(row["do_no_harm_drift_cm"] for row in rows), 4),
    }
    return {
        "what": "known-answer and do-no-harm gates for the support-contact cleaner",
        "note": "the injected tilt is ours, from a seed; the clean reference builds the test and "
        "measures recovery, and is never an input or target for anything run on Tier 2",
        "summary": summary,
        "per_episode": rows,
    }


def save(record: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
