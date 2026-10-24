"""Known-answer tests for the shared geometry.

These are the properties the cleaners' correctness rests on, each one chosen because getting it
wrong would be silent: a reflection reads as a perfect fit, a tilt correction that leaks yaw
looks plausible, and smoothing that shifts a trajectory moves every rest segment.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from v2d.geometry import (
    apply_rigid,
    kabsch,
    minimal_rotation,
    pose_matrices,
    rest_segments,
    rotation_error_deg,
    smooth,
    speed,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- kabsch


def test_kabsch_recovers_a_known_transform_exactly():
    points = RNG.normal(size=(50, 3))
    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    offset = np.array([1.0, -2.0, 0.5])
    fitted_rotation, fitted_offset = kabsch(points, points @ rotation.T + offset)
    assert np.allclose(fitted_rotation, rotation, atol=1e-10)
    assert np.allclose(fitted_offset, offset, atol=1e-10)


def test_kabsch_never_returns_a_reflection():
    """A reflection fits a mirrored cloud perfectly and is not a rigid transform."""
    points = RNG.normal(size=(40, 3))
    mirrored = points * np.array([1.0, 1.0, -1.0])
    rotation, _ = kabsch(points, mirrored)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-10)


def test_kabsch_is_exact_on_a_pure_translation():
    points = RNG.normal(size=(10, 3))
    rotation, offset = kabsch(points, points + np.array([0.0, 0.0, 0.02]))
    assert np.allclose(rotation, np.eye(3), atol=1e-10)
    assert offset == pytest.approx([0.0, 0.0, 0.02], abs=1e-12)


# ---------------------------------------------------------------- minimal_rotation


def test_minimal_rotation_takes_source_onto_target():
    source = np.array([0.3, -0.5, 0.8])
    target = np.array([0.0, 0.0, 1.0])
    rotated = minimal_rotation(source, target).apply(source / np.linalg.norm(source))
    assert np.allclose(rotated, target, atol=1e-12)


def test_minimal_rotation_leaves_yaw_untouched():
    """Its axis must lie in the support plane, or the "tilt only" claim is false."""
    normal = np.array([0.0, 0.0, 1.0])
    facet = np.array([0.2, 0.1, -0.97])
    correction = minimal_rotation(facet, -normal)
    axis = correction.as_rotvec()
    assert np.dot(axis, normal) == pytest.approx(0.0, abs=1e-12)


def test_minimal_rotation_is_identity_when_already_aligned():
    direction = np.array([0.0, 1.0, 0.0])
    assert minimal_rotation(direction, direction).magnitude() == pytest.approx(0.0)


def test_minimal_rotation_handles_antiparallel_without_nan():
    direction = np.array([0.0, 0.0, 1.0])
    correction = minimal_rotation(direction, -direction)
    assert correction.magnitude() == pytest.approx(np.pi)
    assert np.allclose(correction.apply(direction), -direction, atol=1e-12)


# ---------------------------------------------------------------- smoothing and rest


def test_smooth_preserves_a_constant_trajectory():
    constant = np.tile([1.0, 2.0, 3.0], (20, 1))
    assert np.allclose(smooth(constant, 9), constant)


def test_smooth_does_not_shift_a_linear_ramp():
    """Edge padding keeps the interior unbiased; a shift would move every rest segment."""
    ramp = np.stack([np.linspace(0, 1, 60)] * 3, axis=1)
    smoothed = smooth(ramp, 9)
    assert np.allclose(smoothed[10:-10], ramp[10:-10], atol=1e-12)


def test_smooth_window_one_is_the_identity():
    track = RNG.normal(size=(15, 3))
    assert np.allclose(smooth(track, 1), track)


def test_smoothing_cuts_the_apparent_speed_of_a_stationary_noisy_object():
    """The real failure: Tier 2 jitter differentiated at 30 fps swamps a 0.02 m/s threshold.

    The mechanism, not a rest fraction. A rest fraction depends on the noise's frame-to-frame
    correlation -- Tier 2's is a slow perturbation, not white -- so asserting one here would be
    asserting a property of the synthetic noise rather than of the smoothing.
    """
    frames = 600
    resting = RNG.normal(scale=0.0073, size=(frames, 3))
    raw = np.median(speed(resting, 30.0, 1))
    smoothed = np.median(speed(resting, 30.0, 9))
    assert raw > 0.02, "white noise at this scale should defeat the threshold"
    assert smoothed < raw / 2.5


def test_a_slow_drift_is_rest_after_smoothing_and_not_before():
    """Tier 2's position error is nearly constant within a clip, so model it as a drift."""
    frames = 400
    drift = np.cumsum(RNG.normal(scale=0.0004, size=(frames, 3)), axis=0)
    jitter = RNG.normal(scale=0.0025, size=(frames, 3))
    track = drift + jitter
    assert rest_segments(track, 30.0, 0.02, 5, window=1) == []
    assert sum(stop - start for start, stop in rest_segments(track, 30.0, 0.02, 5, 9)) > 0


def test_speed_of_a_known_ramp():
    track = np.stack([np.arange(10) * 0.01, np.zeros(10), np.zeros(10)], axis=1)
    assert speed(track, 30.0, 1)[5] == pytest.approx(0.3)


def test_rest_segments_respects_the_minimum_length():
    moving = np.stack([np.arange(20) * 1.0] * 3, axis=1)
    moving[10:13] = moving[10]  # a 3-frame pause
    assert rest_segments(moving, 30.0, 0.02, 5, 1) == []


# ---------------------------------------------------------------- quaternions


def test_quaternion_reordering_round_trips():
    wxyz = np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]])
    assert np.allclose(xyzw_to_wxyz(wxyz_to_xyzw(wxyz)), wxyz)


def test_wxyz_to_xyzw_moves_the_scalar_last():
    assert wxyz_to_xyzw(np.array([[0.1, 0.2, 0.3, 0.4]])).tolist() == [[0.2, 0.3, 0.4, 0.1]]


def test_pose_matrices_are_homogeneous_and_rotate_correctly():
    position = np.array([[1.0, 2.0, 3.0]])
    quat_wxyz = Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]][None]
    matrices = pose_matrices(position, quat_wxyz)
    assert matrices.shape == (1, 4, 4)
    assert matrices[0, 3].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert np.allclose(matrices[0, :3, 3], position[0])
    assert np.allclose(matrices[0, :3, :3] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_pose_matrices_rejects_a_degenerate_quaternion():
    with pytest.raises(ValueError, match="degenerate"):
        pose_matrices(np.zeros((1, 3)), np.zeros((1, 4)))


def test_rotation_error_of_a_known_angle():
    identity = np.tile(Rotation.identity().as_quat(), (4, 1))
    turned = np.tile(Rotation.from_euler("x", 29, degrees=True).as_quat(), (4, 1))
    assert np.allclose(rotation_error_deg(turned, identity), 29.0, atol=1e-9)


def test_apply_rigid_matches_applying_it_by_hand():
    pose = np.concatenate([RNG.normal(size=(6, 3)), Rotation.random(6, rng=0).as_quat()], axis=1)
    rotation = Rotation.from_euler("y", 0.7).as_matrix()
    offset = np.array([0.1, 0.2, 0.3])
    moved = apply_rigid(pose, rotation, offset)
    assert np.allclose(moved[:, :3], pose[:, :3] @ rotation.T + offset)
    expected = rotation @ Rotation.from_quat(pose[:, 3:]).as_matrix()
    assert np.allclose(Rotation.from_quat(moved[:, 3:]).as_matrix(), expected, atol=1e-12)
