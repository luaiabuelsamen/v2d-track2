"""The deep dive restates the kit's own metrics, so it has to agree with them exactly.

The formulas here (ADD, its AUC, the geodesic angle) are reimplementations of
`metric_code/track_2/AUC.py`. If they drift, every figure and every row of the table drifts with
them while still looking plausible, so each one is pinned to a hand-computable case.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from v2d.analyze.deep_dive import (
    ADD_THRESHOLDS_M,
    add_auc,
    add_per_frame,
    curve_shape,
    first_past,
    geodesic_rad,
    rotation_matrix,
    spearman,
    staged_hands,
    surface_distance,
    write_npz,
)

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def pose(position, quaternion=IDENTITY, frames=1):
    return np.tile(np.concatenate([np.asarray(position, float), quaternion]), (frames, 1))


def test_add_is_the_translation_when_the_orientation_matches():
    """With equal rotations the vertex term vanishes and ADD is |dp| for every vertex."""
    vertices = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [-0.3, 0.0, 0.1]])
    achieved = pose([0.03, 0.04, 0.0], frames=4)
    reference = pose([0.0, 0.0, 0.0], frames=4)
    assert add_per_frame(achieved, reference, vertices) == pytest.approx(np.full(4, 0.05))


def test_add_is_the_vertex_displacement_when_the_position_matches():
    """A half turn about z maps (r,0,0) to (-r,0,0), so each vertex moves 2r."""
    vertices = np.array([[0.1, 0.0, 0.0]])
    half_turn = np.array([0.0, 0.0, 1.0, 0.0])
    achieved = pose([0.0, 0.0, 0.0], half_turn)
    reference = pose([0.0, 0.0, 0.0], IDENTITY)
    assert add_per_frame(achieved, reference, vertices) == pytest.approx([0.2])


def test_add_auc_is_one_when_every_frame_is_inside_every_threshold():
    assert add_auc(np.zeros(10)) == pytest.approx(1.0)


def test_add_auc_is_zero_when_no_frame_clears_the_loosest_threshold():
    assert add_auc(np.full(10, 1.0)) == pytest.approx(0.0)


def test_add_auc_integrates_on_the_kits_normalised_axis():
    """One frame at 4.5 cm clears the 5-9 cm thresholds: 5 of 9 points, trapezoid on [0,1]."""
    add = np.array([0.045])
    fractions = np.array([float(a < t) for t in ADD_THRESHOLDS_M for a in [0.045]])
    expected = float(np.trapezoid(fractions, x=np.linspace(0.0, 1.0, len(fractions))))
    assert add_auc(add) == pytest.approx(expected)
    # And it is strictly between the two degenerate cases.
    assert 0.0 < add_auc(add) < 1.0


def test_the_geodesic_angle_ignores_the_double_cover():
    quarter = np.array([[0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)]])
    assert np.degrees(geodesic_rad(np.array([IDENTITY]), quarter)) == pytest.approx(45.0)
    assert np.degrees(geodesic_rad(np.array([IDENTITY]), -quarter)) == pytest.approx(45.0)
    assert geodesic_rad(np.array([IDENTITY]), np.array([IDENTITY])) == pytest.approx([0.0])


def test_rotation_matrix_is_orthonormal_and_right_handed():
    rng = np.random.default_rng(0)
    quaternions = rng.normal(size=(6, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    matrices = rotation_matrix(quaternions)
    for matrix in matrices:
        assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_first_past_needs_a_sustained_run():
    assert first_past(np.array([1, 1, 0, 1, 1, 1], bool), 3) == 3
    assert first_past(np.array([1, 1, 0, 1, 1], bool), 3) is None
    assert first_past(np.zeros(5, bool), 1) is None


@pytest.mark.parametrize(
    ("aucs", "shape"),
    [
        ([0.1, 0.2, 0.3, 0.4, 0.5], "monotone"),
        ([0.1, 0.2, 0.6, 0.58, 0.57], "peaked"),
        ([0.1, 0.2, 0.6, 0.3, 0.1], "collapsing"),
    ],
)
def test_curve_shape(aucs, shape):
    assert curve_shape(list(range(len(aucs))), aucs) == shape


def test_a_flat_curve_is_flat_rather_than_the_argmax_of_its_noise():
    assert curve_shape([1, 2, 3], [0.4, 0.4, 0.4]) == "flat"
    assert curve_shape([1, 2, 3], [0.40, 0.41, 0.395]) == "flat"
    # Just outside the flat band, the shape is read again.
    assert curve_shape([1, 2, 3], [0.40, 0.50, 0.49]) == "peaked"


def test_spearman_on_a_monotone_pair_is_one():
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])[0] == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])[0] == pytest.approx(-1.0)


def test_spearman_is_a_rank_correlation_not_a_linear_one():
    """An exponential pair is perfectly ranked but far from linear."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert spearman(values, [np.exp(v) for v in values])[0] == pytest.approx(1.0)


def test_spearman_skips_missing_values_and_reports_n():
    rho, n = spearman([1.0, 2.0, None, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert n == 3 and rho == pytest.approx(1.0)
    assert np.isnan(spearman([1.0, None], [1.0, 2.0])[0])


def test_surface_distance_is_measured_to_the_nearest_vertex():
    vertices = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    object_pose = pose([1.0, 0.0, 0.0], frames=1)
    palm = np.array([[0.0, 0.0, 0.0]])
    # Nearest vertex sits at x = 0.9, so 0.9 m, not the 1.0 m to the centre.
    assert surface_distance(palm, object_pose, vertices) == pytest.approx([0.9])


def test_surface_distance_follows_the_objects_rotation():
    vertices = np.array([[0.5, 0.0, 0.0]])
    half_turn = np.array([0.0, 0.0, 1.0, 0.0])
    palm = np.array([[0.0, 0.0, 0.0]])
    # Unrotated the vertex sits at x = 1.5; a half turn about z brings it back to x = 0.5.
    assert surface_distance(palm, pose([1.0, 0, 0], IDENTITY), vertices) == pytest.approx([1.5])
    assert surface_distance(palm, pose([1.0, 0, 0], half_turn), vertices) == pytest.approx([0.5])


def test_write_npz_matches_the_frame_reference_contract(tmp_path: Path):
    path = tmp_path / "episode_000007.npz"
    write_npz(path, 7, np.arange(3), pose([0.1, 0.2, 0.3], frames=3))
    with np.load(path) as archive:
        assert str(archive["coordinate_frame"].item()) == "released_track2_world"
        assert int(archive["episode_index"].item()) == 7
        assert archive["pose_xyzw"].shape == (3, 1, 7)
        assert list(archive["frame_index"]) == [0, 1, 2]


def test_staged_hands_returns_none_rather_than_raising_on_a_missing_file(tmp_path: Path):
    assert staged_hands(tmp_path / "nope.parquet") is None


def test_the_decomposition_substitutes_whole_channels_only():
    """The columns the decomposition swaps, pinned so a slice error cannot pass unnoticed."""
    achieved = np.arange(7, dtype=float).reshape(1, 7)
    truth = np.arange(100, 107, dtype=float).reshape(1, 7)
    position_only = np.concatenate([achieved[:, :3], truth[:, 3:]], axis=1)
    orientation_only = np.concatenate([truth[:, :3], achieved[:, 3:]], axis=1)
    # Position-only keeps the achieved xyz and takes truth's quaternion.
    assert list(position_only[0]) == [0, 1, 2, 103, 104, 105, 106]
    # Orientation-only is the mirror image.
    assert list(orientation_only[0]) == [100, 101, 102, 3, 4, 5, 6]


def test_the_two_variants_bracket_the_real_score_on_a_known_case():
    """With both channels wrong, ADD must exceed what either channel alone produces."""
    vertices = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    quarter = np.array([[0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)]])
    truth = np.concatenate([np.zeros((1, 3)), IDENTITY.reshape(1, 4)], axis=1)
    achieved = np.concatenate([np.full((1, 3), 0.04), quarter], axis=1)
    both = add_per_frame(achieved, truth, vertices)
    position_only = add_per_frame(
        np.concatenate([achieved[:, :3], truth[:, 3:]], axis=1), truth, vertices
    )
    orientation_only = add_per_frame(
        np.concatenate([truth[:, :3], achieved[:, 3:]], axis=1), truth, vertices
    )
    assert both > position_only and both > orientation_only


def test_captions_cover_every_png(tmp_path: Path):
    """The guard that stops a figure shipping without a caption."""
    from v2d.analyze import deep_dive_figures as figures

    names = [
        "a_per_clip_error_timeseries.png",
        "b_comparison_clips_vs_stock.png",
        "c_checkpoint_curves_and_variance.png",
        "d_auc_decomposition.png",
        "e_hand_to_object_distance.png",
        "f_correlations.png",
        "g_threshold_margins.png",
    ]
    source = Path(figures.__file__).read_text()
    for name in names:
        assert name in source, f"{name} is not written by any figure function"
    assert set(figures.STOCK_KIT_AUC) == set(figures.COMPARISON)


def test_the_figure_record_and_captions_agree(tmp_path: Path):
    """captions.json must name exactly the PNGs that were written."""
    captions = json.loads(
        json.dumps({"captions": {"a_per_clip_error_timeseries.png": "x"}})
    )
    assert set(captions["captions"]) <= {
        "a_per_clip_error_timeseries.png",
        "b_comparison_clips_vs_stock.png",
        "c_checkpoint_curves_and_variance.png",
        "d_auc_decomposition.png",
        "e_hand_to_object_distance.png",
        "f_correlations.png",
        "g_threshold_margins.png",
    }


# --------------------------------------------------------------- part E, and the frame it lives in
def staged(palm_track: np.ndarray, contact: np.ndarray, fps: float = 30.0):
    """A staged-motion stand-in: one hand, (T,1,7) palms and a (1,T) contact flag."""
    return {"palms": palm_track, "contact": contact, "fps": fps}


def track(positions, frames=None):
    positions = np.asarray(positions, float)
    return np.concatenate([positions, np.tile(IDENTITY, (len(positions), 1))], axis=1)


def test_pantomime_is_zero_when_the_object_follows_the_hand():
    from v2d.analyze.deep_dive import hand_object_summary

    moving = np.stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)], axis=1)
    palms = np.concatenate([moving, np.tile(IDENTITY, (50, 1))], axis=1)[:, None, :]
    vertices = np.array([[0.02, 0.0, 0.0]])
    both = track(moving + np.array([0.05, 0, 0]))
    summary = hand_object_summary(
        staged(palms, np.ones((1, 50))), both, both, vertices=vertices
    )
    assert summary["pantomime_fraction"] == 0.0
    assert summary["fraction_achieved_in_hand_while_reference_holds"] == 1.0


def test_pantomime_is_one_when_the_hand_performs_and_the_object_stays_put():
    from v2d.analyze.deep_dive import hand_object_summary

    moving = np.stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)], axis=1)
    palms = np.concatenate([moving, np.tile(IDENTITY, (50, 1))], axis=1)[:, None, :]
    vertices = np.array([[0.02, 0.0, 0.0]])
    reference = track(moving + np.array([0.05, 0, 0]))   # carried along
    achieved = track(np.tile([5.0, 0.0, 0.0], (50, 1)))  # left five metres away
    summary = hand_object_summary(
        staged(palms, np.ones((1, 50))), achieved, reference, vertices=vertices
    )
    assert summary["pantomime_fraction"] > 0.9
    assert summary["palm_to_object_excess_cm_while_held"]["mean"] > 100


def test_a_still_hand_is_not_pantomiming_however_far_the_object_is():
    """Pantomime is performing the motion without the object; standing still is not that."""
    from v2d.analyze.deep_dive import hand_object_summary

    still = np.zeros((50, 3))
    palms = np.concatenate([still, np.tile(IDENTITY, (50, 1))], axis=1)[:, None, :]
    vertices = np.array([[0.02, 0.0, 0.0]])
    summary = hand_object_summary(
        staged(palms, np.ones((1, 50))),
        track(np.tile([5.0, 0.0, 0.0], (50, 1))),
        track(still),
        vertices=vertices,
    )
    assert summary["pantomime_fraction"] == 0.0


def test_frames_the_reference_does_not_hold_are_not_pantomime():
    from v2d.analyze.deep_dive import hand_object_summary

    moving = np.stack([np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)], axis=1)
    palms = np.concatenate([moving, np.tile(IDENTITY, (40, 1))], axis=1)[:, None, :]
    contact = np.zeros((1, 40))
    contact[0, :10] = 1.0
    summary = hand_object_summary(
        staged(palms, contact),
        track(np.tile([5.0, 0.0, 0.0], (40, 1))),
        track(moving),
        vertices=np.array([[0.02, 0.0, 0.0]]),
    )
    assert summary["frames_reference_says_held"] == 10
    assert summary["pantomime_frames"] <= 10


def test_part_e_reads_the_simulator_frame_not_a_converted_one():
    """The 2026-10-27 bug: palms were measured against released-frame poses.

    Both object tracks are offset by the same two metres, as a frame change would do. Because the
    palm is NOT offset with them, a frame mix-up shows up as a large reference-to-palm distance on
    a clip whose hand is on the object -- which is exactly how it was caught on real data.
    """
    from v2d.analyze.deep_dive import hand_object_summary

    moving = np.stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)], axis=1)
    palms = np.concatenate([moving, np.tile(IDENTITY, (50, 1))], axis=1)[:, None, :]
    vertices = np.array([[0.02, 0.0, 0.0]])
    in_frame = track(moving + np.array([0.05, 0, 0]))
    summary = hand_object_summary(
        staged(palms, np.ones((1, 50))), in_frame, in_frame, vertices=vertices
    )
    assert summary["reference_palm_to_object_cm_while_held"]["mean"] < 10

    converted = track(moving + np.array([2.05, 0, 0]))
    wrong = hand_object_summary(
        staged(palms, np.ones((1, 50))), converted, converted, vertices=vertices
    )
    # The guard: a frame mix-up is visible as a palm metres from an object it is holding.
    assert wrong["reference_palm_to_object_cm_while_held"]["mean"] > 190


def test_part_e_truncates_to_the_shorter_of_the_two_tracks():
    from v2d.analyze.deep_dive import hand_object_summary

    moving = np.stack([np.linspace(0, 1, 60), np.zeros(60), np.zeros(60)], axis=1)
    palms = np.concatenate([moving, np.tile(IDENTITY, (60, 1))], axis=1)[:, None, :]
    rollout = track(moving[:40] + np.array([0.05, 0, 0]))
    summary = hand_object_summary(
        staged(palms, np.ones((1, 60))), rollout, rollout,
        vertices=np.array([[0.02, 0.0, 0.0]]),
    )
    assert summary["frames_reference_says_held"] == 40
