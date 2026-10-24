"""Readers for the released Track 2 dataset.

One place that knows the release's conventions, so no caller has to remember them:

- `observation.object.pose` is position followed by a **wxyz** quaternion; everything downstream
  of this module speaks xyzw.
- `observation.object.visible` has exactly one False frame in the whole release (episode 11's
  last), and that frame is zero-filled. A zero quaternion reads back as *identity at the world
  origin*, which is one phantom point capable of dragging a global rigid fit to a 113 mm
  residual. It is also the frame the organisers omit from the scored grid.
- `observation.human.pose` is ``(T, 77*3)`` local rotation vectors with the root first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .config import TIER1, TIER_DIRS
from .geometry import wxyz_to_xyzw


@dataclass(frozen=True)
class Episode:
    """One row of `meta/episodes_metadata.jsonl`, with the fields this project uses."""

    index: int
    sequence_id: str
    object_name: str
    frames: int
    mesh: str
    ground_plane: str
    support_surface: str
    task: str


def tier_root(dataset_root: Path, tier: str) -> Path:
    """The directory holding one tier's `data/`, `meta/`, `mesh/` and friends."""
    return dataset_root / "track_2" / TIER_DIRS[tier]


def episode_parquet(dataset_root: Path, tier: str, episode: int) -> Path:
    return tier_root(dataset_root, tier) / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"


def episodes(dataset_root: Path) -> list[Episode]:
    """Every scored episode, ordered by index.

    Meshes, ground planes and support surfaces ship identically under each tier, so they are
    always read from Tier 1.
    """
    path = dataset_root / "track_2" / TIER1 / "meta" / "episodes_metadata.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["episode_index"])
    return [
        Episode(
            index=int(row["episode_index"]),
            sequence_id=row["sequence_id"],
            object_name=row["object"],
            frames=int(row["frames"]),
            mesh=row["mesh"],
            ground_plane=row["ground_plane"],
            support_surface=row["support_surface"],
            task=row.get("task", ""),
        )
        for row in rows
    ]


def object_pose(dataset_root: Path, tier: str, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """``(T,7)`` xyz + xyzw object pose and the ``(T,)`` visibility mask."""
    table = pq.read_table(
        episode_parquet(dataset_root, tier, episode),
        columns=["observation.object.pose", "observation.object.visible"],
    )
    pose = np.asarray(table["observation.object.pose"].to_pylist(), dtype=np.float64)
    visible = np.asarray(table["observation.object.visible"].to_pylist(), dtype=bool)
    return np.concatenate([pose[:, :3], wxyz_to_xyzw(pose[:, 3:])], axis=1), visible


def human_params(dataset_root: Path, tier: str, episode: int) -> dict[str, np.ndarray]:
    """The SOMA-X human fields, with `pose` reshaped to ``(T, 77, 3)``."""
    table = pq.read_table(episode_parquet(dataset_root, tier, episode))
    pose = np.asarray(table["observation.human.pose"].to_pylist(), dtype=np.float64)
    return {
        "pose": pose.reshape(len(pose), -1, 3),
        "translation": np.asarray(
            table["observation.human.translation"].to_pylist(), dtype=np.float64
        ),
        "identity_coeffs": np.asarray(
            table["observation.human.identity_coeffs"].to_pylist(), dtype=np.float64
        ),
        "scale_params": np.asarray(
            table["observation.human.scale_params"].to_pylist(), dtype=np.float64
        ),
        "bone_length_flexibles": np.asarray(
            table["observation.human.bone_length_flexibles"].to_pylist(), dtype=np.float64
        ),
        "visible": np.asarray(table["observation.object.visible"].to_pylist(), dtype=bool),
    }


def hold_last_visible(pose: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, int]:
    """Carry the last visible pose across invisible frames; also return how many were filled.

    Without this an invisible frame's zero-filled quaternion becomes the identity rotation at the
    world origin, which is a phantom observation rather than missing data.
    """
    if visible.all():
        return pose, 0
    index = np.arange(len(visible))
    index[~visible] = -1
    index = np.maximum.accumulate(index)
    if index[0] < 0:
        index[index < 0] = int(np.argmax(visible)) if visible.any() else 0
    return pose[index], int((~visible).sum())


def ground_plane(dataset_root: Path, episode: Episode) -> tuple[np.ndarray, float]:
    """The released floor plane as a unit normal and a signed offset."""
    path = tier_root(dataset_root, "tier1") / episode.ground_plane
    plane = np.asarray(json.loads(path.read_text())["plane"], dtype=np.float64)
    norm = np.linalg.norm(plane[:3])
    return plane[:3] / norm, float(plane[3] / norm)


def oriented_ground_plane(
    dataset_root: Path, episode: Episode, trajectory: np.ndarray
) -> tuple[np.ndarray, float]:
    """The floor plane, flipped so the object sits on its positive side.

    The sign is decided from the *noisy* trajectory alone, so this stays usable by a deployable
    cleaner that never sees the clean reference.
    """
    normal, offset = ground_plane(dataset_root, episode)
    if np.mean(trajectory[:, :3] @ normal + offset) < 0:
        return -normal, -offset
    return normal, offset


def object_hull(
    dataset_root: Path, episode: Episode, max_facets: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Convex-hull vertices and the outward normals of its largest facets.

    Tiny facets cannot support an object, so trimming to the largest keeps the support-facet
    search both cheap and sane.
    """
    import trimesh

    path = tier_root(dataset_root, "tier1") / episode.mesh
    mesh = trimesh.load(str(path), force="mesh", process=False)
    hull = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)
    ).convex_hull
    normals = np.asarray(hull.face_normals, dtype=np.float64)
    if len(normals) > max_facets:
        keep = np.argsort(np.asarray(hull.area_faces, dtype=np.float64))[::-1][:max_facets]
        normals = normals[keep]
    return np.asarray(hull.vertices, dtype=np.float64), normals
