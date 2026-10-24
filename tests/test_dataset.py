"""Known-answer tests for the release's conventions.

The invisible-frame handling has a real failure behind it: episode 11's last frame is the one
invisible frame in the whole release, it is zero-filled, and a zero quaternion reads back as
*identity at the world origin*. That single phantom point dragged a global rigid fit to a 113 mm
residual and faked a 24-36 degree orientation error.
"""

from __future__ import annotations

import numpy as np

from v2d.dataset import hold_last_visible


def test_all_visible_is_a_passthrough():
    pose = np.arange(20, dtype=np.float64).reshape(4, 5)
    held, filled = hold_last_visible(pose, np.ones(4, dtype=bool))
    assert filled == 0
    assert held is pose


def test_a_trailing_invisible_frame_takes_the_last_visible_pose():
    pose = np.array([[1.0], [2.0], [3.0], [0.0]])
    visible = np.array([True, True, True, False])
    held, filled = hold_last_visible(pose, visible)
    assert filled == 1
    assert held[:, 0].tolist() == [1.0, 2.0, 3.0, 3.0]


def test_a_zero_filled_row_never_survives():
    """The actual hazard: a zeroed row is identity-at-origin, not missing data."""
    pose = np.zeros((3, 7))
    pose[0] = [1, 2, 3, 0, 0, 0, 1]
    pose[1] = [1, 2, 3, 0, 0, 0, 1]
    visible = np.array([True, True, False])
    held, _ = hold_last_visible(pose, visible)
    assert not np.allclose(held[2], np.zeros(7))
    assert np.allclose(held[2], pose[1])


def test_a_leading_invisible_run_borrows_the_first_visible_pose():
    pose = np.array([[0.0], [0.0], [7.0], [8.0]])
    visible = np.array([False, False, True, True])
    held, filled = hold_last_visible(pose, visible)
    assert filled == 2
    assert held[:, 0].tolist() == [7.0, 7.0, 7.0, 8.0]


def test_an_interior_gap_is_bridged_forward():
    pose = np.array([[1.0], [0.0], [0.0], [4.0]])
    visible = np.array([True, False, False, True])
    held, filled = hold_last_visible(pose, visible)
    assert filled == 2
    assert held[:, 0].tolist() == [1.0, 1.0, 1.0, 4.0]


def test_nothing_visible_at_all_does_not_crash():
    pose = np.zeros((3, 2))
    held, filled = hold_last_visible(pose, np.zeros(3, dtype=bool))
    assert filled == 3
    assert held.shape == pose.shape
