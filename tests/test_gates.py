"""Tests for the synthetic gate and the per-clip selection it drives.

`gate_passing` is now the only rule in the pipeline that decides *where* a correction is
applied, so it is worth more than a glance. The injection has to be a genuine tilt, because a
recovery figure measured against a rotation the cleaner was never built to fix would be
meaningless in both directions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from v2d.clean.gates import Injection
from v2d.clean.support_contact import gate_passing

NORMAL = np.array([0.0, 0.0, 1.0])


def _record(tmp_path, rows):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"per_episode": rows}))
    return path


# ---------------------------------------------------------------- selection


def test_only_clips_clearing_the_bar_are_selected(tmp_path):
    path = _record(
        tmp_path,
        [
            {"episode": 1, "recovered_fraction": 0.9},
            {"episode": 2, "recovered_fraction": 0.5},
            {"episode": 3, "recovered_fraction": 0.49},
            {"episode": 4, "recovered_fraction": -0.3},
        ],
    )
    assert gate_passing(path, 0.50) == {1, 2}


def test_a_clip_with_no_recovery_figure_is_not_selected(tmp_path):
    """A clip with no rest frames has nothing to recover, so it is not evidence of success."""
    path = _record(
        tmp_path,
        [{"episode": 1, "recovered_fraction": None}, {"episode": 2, "recovered_fraction": 0.8}],
    )
    assert gate_passing(path, 0.50) == {2}


def test_the_bar_is_inclusive(tmp_path):
    path = _record(tmp_path, [{"episode": 7, "recovered_fraction": 0.5}])
    assert gate_passing(path, 0.5) == {7}


def test_a_bar_nothing_clears_selects_nothing(tmp_path):
    path = _record(tmp_path, [{"episode": 1, "recovered_fraction": 0.3}])
    assert gate_passing(path, 0.99) == set()


def test_the_committed_gate_record_selects_the_seven_clips_we_shipped():
    """Guards the number quoted in docs/GATES.md and the README against a silent drift."""
    from pathlib import Path

    record = Path("results/gate_cleaner_table.json")
    if not record.is_file():
        pytest.skip("gate record not present")
    assert gate_passing(record, 0.50) == {2, 3, 9, 16, 18, 24, 25}


# ---------------------------------------------------------------- the injection


def test_the_injected_rotation_is_a_pure_tilt():
    """Its axis must lie in the support plane. A component along the normal would be yaw, which
    no support-contact argument can recover, so including it would rig the gate to fail."""
    pose = np.concatenate(
        [np.zeros((5, 3)), np.tile(Rotation.identity().as_quat(), (5, 1))], axis=1
    )
    _, injected = Injection(tilt_deg=12.0, seed=0).apply(pose, NORMAL, np.random.default_rng(0))
    assert float(np.dot(injected.as_rotvec(), NORMAL)) == pytest.approx(0.0, abs=1e-12)


def test_the_injected_magnitude_is_what_was_asked_for():
    pose = np.concatenate(
        [np.zeros((3, 3)), np.tile(Rotation.identity().as_quat(), (3, 1))], axis=1
    )
    _, injected = Injection(tilt_deg=12.0).apply(pose, NORMAL, np.random.default_rng(1))
    assert np.degrees(injected.magnitude()) == pytest.approx(12.0)


def test_the_injection_leaves_position_alone():
    """Rest detection is position-based, so moving positions would change which frames are rest
    and the before/after comparison would no longer be paired."""
    rng = np.random.default_rng(2)
    pose = np.concatenate([rng.normal(size=(8, 3)), Rotation.random(8, rng=3).as_quat()], axis=1)
    noisy, _ = Injection().apply(pose, NORMAL, np.random.default_rng(4))
    assert np.array_equal(noisy[:, :3], pose[:, :3])


def test_the_sign_varies_with_the_seed():
    """A cleaner must not be able to pass by always tilting one way."""
    pose = np.concatenate(
        [np.zeros((2, 3)), np.tile(Rotation.identity().as_quat(), (2, 1))], axis=1
    )
    # With normal=+z the axis is cross(+z, +x) = +y, so that is the component to read.
    signs = set()
    for seed in range(12):
        _, injected = Injection().apply(pose, NORMAL, np.random.default_rng(seed))
        signs.add(float(np.sign(injected.as_rotvec()[1])))
    assert signs == {-1.0, 1.0}, f"both tilt directions should occur across seeds, saw {signs}"


def test_the_injection_is_reproducible_from_its_seed():
    pose = np.concatenate(
        [np.zeros((4, 3)), np.tile(Rotation.identity().as_quat(), (4, 1))], axis=1
    )
    first, _ = Injection(seed=5).apply(pose, NORMAL, np.random.default_rng(5))
    second, _ = Injection(seed=5).apply(pose, NORMAL, np.random.default_rng(5))
    assert np.array_equal(first, second)


def test_a_normal_along_x_still_gives_a_perpendicular_axis():
    """The helper-vector fallback has to hold for any support orientation, not just +z."""
    normal = np.array([1.0, 0.0, 0.0])
    pose = np.concatenate(
        [np.zeros((2, 3)), np.tile(Rotation.identity().as_quat(), (2, 1))], axis=1
    )
    _, injected = Injection().apply(pose, normal, np.random.default_rng(0))
    assert float(np.dot(injected.as_rotvec(), normal)) == pytest.approx(0.0, abs=1e-12)
