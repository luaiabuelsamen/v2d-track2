"""Tests for the rollout frame conversion.

The failure this guards against is severe and quiet: scoring a rollout in the evaluator's own
frame read MPPE 306.74 cm where the correct answer was 14.74 cm. A 20x error that looks like a
catastrophically bad policy rather than a frame mismatch, and nothing in the numbers says which.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from v2d.submit.rollout import ANGLE_TOLERANCE_RAD, POSITION_TOLERANCE_M, convert

RNG = np.random.default_rng(0)


def _poses(count: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(size=(count, 3)), Rotation.random(count, rng=seed).as_quat()], axis=1
    )


def _apply(pose: np.ndarray, rotation: Rotation, offset: np.ndarray) -> np.ndarray:
    out = np.empty_like(pose)
    out[:, :3] = rotation.apply(pose[:, :3]) + offset
    out[:, 3:] = (rotation * Rotation.from_quat(pose[:, 3:])).as_quat()
    return out


def test_a_known_transform_is_recovered_and_verified():
    """The reference residual is the gate: on a correct rollout it is essentially zero."""
    evaluator_reference = _poses(40)
    rotation = Rotation.from_euler("xyz", [0.4, -1.1, 2.3])
    offset = np.array([3.0, -1.0, 0.5])
    released = _apply(evaluator_reference, rotation, offset)
    # the policy tracked its reference imperfectly, in the evaluator's frame
    achieved = evaluator_reference.copy()
    achieved[:, :3] += 0.05

    converted, position_residual, angle_residual = convert(achieved, evaluator_reference, released)
    assert position_residual < POSITION_TOLERANCE_M
    assert angle_residual < ANGLE_TOLERANCE_RAD
    # the tracking error survives the conversion unchanged, because it is a rigid map
    assert np.allclose(
        np.linalg.norm(converted[:, :3] - released[:, :3], axis=1), 0.05 * np.sqrt(3), atol=1e-9
    )


def test_skipping_the_conversion_is_catastrophic_and_silent():
    """Why the tool exists: the raw achieved pose is simply in the wrong place."""
    evaluator_reference = _poses(30, seed=2)
    rotation = Rotation.from_euler("z", 2.0)
    offset = np.array([5.0, 0.0, 0.0])
    released = _apply(evaluator_reference, rotation, offset)
    achieved = evaluator_reference.copy()

    unconverted = np.linalg.norm(achieved[:, :3] - released[:, :3], axis=1).mean()
    converted, _, _ = convert(achieved, evaluator_reference, released)
    properly = np.linalg.norm(converted[:, :3] - released[:, :3], axis=1).mean()
    assert unconverted > 1.0
    assert properly < 1e-9
    assert unconverted / max(properly, 1e-12) > 1000


def test_a_mismatched_reference_fails_the_check_rather_than_passing_quietly():
    """If the evaluator's reference is not a rigid image of the released input, say so."""
    evaluator_reference = _poses(25, seed=3)
    released = _poses(25, seed=4)  # unrelated
    _, position_residual, angle_residual = convert(
        evaluator_reference, evaluator_reference, released
    )
    assert position_residual > POSITION_TOLERANCE_M
    assert angle_residual > ANGLE_TOLERANCE_RAD


def test_the_conversion_is_exact_on_the_identity():
    reference = _poses(12, seed=5)
    converted, position_residual, angle_residual = convert(reference, reference, reference)
    assert np.allclose(converted, reference, atol=1e-12)
    assert position_residual == pytest.approx(0.0, abs=1e-12)
    assert angle_residual == pytest.approx(0.0, abs=1e-12)


def test_orientation_error_is_preserved_by_the_conversion():
    """A rigid map cannot change the angle between achieved and reference."""
    evaluator_reference = _poses(20, seed=6)
    achieved = evaluator_reference.copy()
    tilt = Rotation.from_euler("x", np.radians(12.0))
    achieved[:, 3:] = (tilt * Rotation.from_quat(achieved[:, 3:])).as_quat()
    rotation = Rotation.from_euler("y", 1.7)
    released = _apply(evaluator_reference, rotation, np.array([1.0, 2.0, 3.0]))

    converted, _, _ = convert(achieved, evaluator_reference, released)
    angles = np.degrees(
        (
            Rotation.from_quat(converted[:, 3:]) * Rotation.from_quat(released[:, 3:]).inv()
        ).magnitude()
    )
    assert np.allclose(angles, 12.0, atol=1e-8)
