"""Known-answer tests for `v2d noise-channels`.

The two measurements that carry a conclusion are tested against values computed by hand: the
palm-frame grasp spread (part B), which decides whether the hands can anchor the object, and the
tilt/yaw split (part A), which decides which half of the object orientation noise matters.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from v2d.analyze.noise_channels import (
    _mean_quat,
    _spearman,
    quat_conjugate,
    quat_multiply,
    temporal,
    tilt_yaw_split,
)
from v2d.clean.tilt_yaw import geodesic_deg, rotation_matrix


def palm_frame_spread(palm_p, palm_q, obj_p, obj_q):
    """The part B measurement, isolated: object pose in the palm frame, spread about its mean."""
    rel_p = np.einsum("tji,tj->ti", rotation_matrix(palm_q), obj_p - palm_p)
    rel_q = quat_multiply(quat_conjugate(palm_q), obj_q)
    spread_cm = float(np.sqrt(rel_p.var(axis=0).sum()) * 100.0)
    mean_q = _mean_quat(rel_q)
    spread_deg = float(geodesic_deg(np.broadcast_to(mean_q, rel_q.shape), rel_q).mean())
    return spread_cm, spread_deg


def test_a_rigid_grasp_measures_zero_spread():
    """A perfectly rigid grasp must read 0 cm and 0 deg however wildly the palm moves."""
    rng = np.random.default_rng(0)
    frames = 200
    palm_rot = Rotation.from_quat(rng.normal(size=(frames, 4)) / 1.0)  # normalized internally
    palm_q = palm_rot.as_quat()
    palm_p = rng.normal(scale=0.5, size=(frames, 3))
    # The object sits at a FIXED offset and a FIXED rotation in the palm frame.
    offset_in_palm = np.array([0.03, -0.01, 0.07])
    rotation_in_palm = Rotation.from_euler("xyz", [11.0, -23.0, 47.0], degrees=True)
    obj_p = palm_p + palm_rot.apply(offset_in_palm)
    obj_q = (palm_rot * rotation_in_palm).as_quat()

    spread_cm, spread_deg = palm_frame_spread(palm_p, palm_q, obj_p, obj_q)
    assert spread_cm == pytest.approx(0.0, abs=1e-6)
    assert spread_deg == pytest.approx(0.0, abs=1e-6)


def test_a_known_slip_measures_its_own_size():
    """A grasp that slips by a known amount reads back that amount."""
    frames = 2
    palm_q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frames, 1))
    palm_p = np.zeros((frames, 3))
    # The object is 4 cm away on x in one frame and 4 cm the other way in the next: the per-axis
    # std is 4 cm, so the total positional spread is 4 cm.
    obj_p = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]])
    # It also rotates 30 deg about z between the two frames, so each frame sits 15 deg off the mean.
    obj_q = Rotation.from_euler("z", [[0.0], [30.0]], degrees=True).as_quat()

    spread_cm, spread_deg = palm_frame_spread(palm_p, palm_q, obj_p, obj_q)
    assert spread_cm == pytest.approx(4.0, abs=1e-6)
    assert spread_deg == pytest.approx(15.0, abs=1e-4)


def test_tilt_yaw_pure_yaw_has_no_tilt():
    """A rotation about the object's own z axis is all yaw and no tilt."""
    clean = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))
    noisy = Rotation.from_euler("z", [[10.0], [-25.0], [40.0]], degrees=True).as_quat()
    tilt, yaw = tilt_yaw_split(clean, noisy)
    assert tilt == pytest.approx(np.zeros(3), abs=1e-6)
    assert yaw == pytest.approx(np.array([10.0, 25.0, 40.0]), abs=1e-4)


def test_tilt_yaw_pure_tilt_has_no_yaw():
    """A rotation about an axis perpendicular to z is all tilt, and the yaw residual vanishes."""
    clean = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))
    noisy = Rotation.from_euler("x", [[10.0], [25.0], [40.0]], degrees=True).as_quat()
    tilt, yaw = tilt_yaw_split(clean, noisy)
    assert tilt == pytest.approx(np.array([10.0, 25.0, 40.0]), abs=1e-4)
    assert yaw == pytest.approx(np.zeros(3), abs=1e-4)


def test_temporal_separates_jitter_from_drift():
    """White noise is nearly all jitter with no autocorrelation; a ramp is all drift."""
    rng = np.random.default_rng(1)
    white = rng.normal(size=4000)
    ramp = np.linspace(0.0, 10.0, 4000)
    assert temporal(white)["lag1_autocorr"] == pytest.approx(0.0, abs=0.05)
    assert temporal(white)["drift_std"] < 0.4 * temporal(white)["jitter_std"]
    assert temporal(ramp)["lag1_autocorr"] == pytest.approx(1.0, abs=0.01)
    assert temporal(ramp)["jitter_std"] == pytest.approx(0.0, abs=1e-9)


def test_spearman_matches_scipy_on_a_known_pair():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _spearman(a, 2.0 * a) == pytest.approx(1.0)
    assert _spearman(a, -a) == pytest.approx(-1.0)
