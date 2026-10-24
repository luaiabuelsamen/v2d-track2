"""Rigid-body geometry shared by the cleaners, the validator and the submission packer.

Every helper here was duplicated across two or more of the old `scripts/*.py` tools, because
those ran under `python -I` (which implies `-P`, so a script's own directory is not on
`sys.path`) and could not import from each other. As an installed package they can, so each of
these now exists once.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def smooth(position: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding, so differentiating does not amplify noise.

    Tier 2's position jitter is ~0.7 cm within a clip. Differentiated at 30 fps that is
    ~0.2 m/s, an order of magnitude above any plausible rest threshold, which leaves a
    stationary object looking like a moving one. Smoothing first is what makes rest detectable.
    """
    if window <= 1:
        return position
    pad = window // 2
    padded = np.pad(position, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")[: len(position)]
            for axis in range(position.shape[1])
        ],
        axis=1,
    )


def speed(position: np.ndarray, fps: float, window: int) -> np.ndarray:
    """Per-frame speed of a smoothed trajectory, in metres per second."""
    return np.linalg.norm(np.gradient(smooth(position, window), axis=0), axis=-1) * fps


def slow_runs(slow: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs of True at least `min_frames` long."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(slow):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_frames:
                runs.append((start, index))
            start = None
    if start is not None and len(slow) - start >= min_frames:
        runs.append((start, len(slow)))
    return runs


def rest_segments(
    position: np.ndarray, fps: float, speed_limit: float, min_frames: int, window: int = 1
) -> list[tuple[int, int]]:
    """Runs where the trajectory moves slower than `speed_limit` metres per second."""
    return slow_runs(speed(position, fps, window) < speed_limit, min_frames)


def quantile_rest_segments(
    position: np.ndarray, fps: float, quantile: float, min_frames: int, window: int
) -> tuple[list[tuple[int, int]], float]:
    """Runs among this trajectory's own quietest `quantile` of frames, plus the speed that was.

    A per-clip quantile where an absolute threshold cannot work. For the human root the Tier 2
    noise floor is 6.2x Tier 1's (0.0330 against 0.0053 m/s), so no fixed speed recovers Tier 1's
    stance set from Tier 2 data, while the *ordering* of quiet frames barely changes.
    """
    values = speed(position, fps, window)
    limit = float(np.quantile(values, quantile))
    return slow_runs(values <= limit, min_frames), limit


def minimal_rotation(source: np.ndarray, target: np.ndarray) -> Rotation:
    """The minimal rotation taking unit vector `source` onto unit vector `target`.

    Its axis is ``source x target``, perpendicular to both. When `target` is a support normal the
    axis lies in the support plane, so applying it tilts the object without rotating it about
    that normal: a pure tilt correction that leaves yaw untouched by construction.
    """
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    axis = np.cross(source, target)
    norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if norm < 1e-12:
        if dot > 0:
            return Rotation.identity()
        # Antiparallel: any perpendicular axis gives a 180 degree turn. Pick a stable one.
        helper = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi)
    return Rotation.from_rotvec(axis / norm * np.arccos(dot))


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rigid ``(rotation, offset)`` minimising ``|rotation @ source + offset - target|``.

    The reflection guard matters: without it a near-degenerate point set can fit an improper
    rotation, which reads as a perfect fit and is not a rigid transform.
    """
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, target_mean - rotation @ source_mean


def apply_rigid(pose: np.ndarray, rotation: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to ``(T, 7)`` poses holding xyz followed by an xyzw quaternion."""
    out = np.empty_like(pose)
    out[:, :3] = pose[:, :3] @ rotation.T + offset
    out[:, 3:] = Rotation.from_matrix(
        rotation @ Rotation.from_quat(pose[:, 3:]).as_matrix()
    ).as_quat()
    return out


def wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Reorder ``(..., 4)`` quaternions from wxyz (the release's order) to xyzw (scipy's)."""
    return quaternion[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Reorder ``(..., 4)`` quaternions from xyzw back to wxyz."""
    return quaternion[..., [3, 0, 1, 2]]


def pose_matrices(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """``(T,3)`` + ``(T,4)`` wxyz to ``(T,4,4)`` homogeneous transforms.

    `soma_to_g1.py` consumes `poses.npy` as ``einsum("ij,tjk->tik", diag(1,-1,-1,1), poses)``, so
    it must hold 4x4 matrices rather than 7-vectors.
    """
    norm = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("degenerate object quaternion")
    out = np.zeros((len(position), 4, 4), dtype=np.float64)
    out[:, :3, :3] = Rotation.from_quat(wxyz_to_xyzw(quat_wxyz / norm)).as_matrix()
    out[:, :3, 3] = position
    out[:, 3, 3] = 1.0
    return out


def rotation_error_deg(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Per-frame angle between two xyzw quaternion trajectories, in degrees."""
    relative = Rotation.from_quat(a_xyzw) * Rotation.from_quat(b_xyzw).inv()
    return np.degrees(relative.magnitude())
