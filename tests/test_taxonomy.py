"""The classifier decides what the next run is aimed at, so each rule gets a case of its own.

The ordering is the part worth pinning: a clip can be at its reference ceiling *and* show a late
loss, or topple *and* then drop, and reporting the second mechanism would point the next run at
the wrong fix.
"""

from __future__ import annotations

import numpy as np

from v2d.analyze.taxonomy import DROP_Z_M, classify, closure_of, drop_frame, summarise


def pose(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    frames = len(x)
    out = np.zeros((frames, 7))
    out[:, 0], out[:, 2], out[:, 6] = x, z, 1.0
    return out


def case(
    *,
    position_cm: np.ndarray,
    orientation_deg: np.ndarray | None = None,
    achieved_x: np.ndarray | None = None,
    achieved_z: np.ndarray | None = None,
    reference_x: np.ndarray | None = None,
    reference_z: np.ndarray | None = None,
    kit_auc: float = 0.1,
    ceiling: float | None = 0.6,
) -> dict:
    frames = len(position_cm)
    zeros, ones = np.zeros(frames), np.full(frames, 0.9)
    return classify(
        position_cm=position_cm,
        orientation_deg=np.zeros(frames) if orientation_deg is None else orientation_deg,
        achieved=pose(
            zeros if achieved_x is None else achieved_x, ones if achieved_z is None else achieved_z
        ),
        reference=pose(
            zeros if reference_x is None else reference_x,
            ones if reference_z is None else reference_z,
        ),
        kit_auc=kit_auc,
        ceiling=ceiling,
    )


def test_a_clip_at_its_ceiling_is_not_reported_as_a_policy_failure():
    """It checks first on purpose: e13 sits 0.02 under its ceiling and shows a late loss too."""
    position = np.concatenate([np.full(80, 2.0), np.full(20, 40.0)])
    verdict = case(position_cm=position, kit_auc=0.16, ceiling=0.18)
    assert verdict["failure_class"] == "reference_ceiling"
    assert verdict["failure_frame"] is None


def test_an_object_that_never_moves_while_the_reference_does_is_never_grasped():
    verdict = case(
        position_cm=np.full(200, 60.0),
        achieved_x=np.zeros(200),
        reference_x=np.linspace(0.0, 1.0, 200),
    )
    assert verdict["failure_class"] == "never_grasped"
    assert verdict["failure_frame"] is not None
    assert verdict["failure_frame"] < 50


def test_an_object_the_policy_moves_is_not_never_grasped():
    verdict = case(
        position_cm=np.full(200, 60.0),
        achieved_x=np.linspace(0.0, 0.5, 200),
        reference_x=np.linspace(0.0, 1.0, 200),
    )
    assert verdict["failure_class"] != "never_grasped"


def test_the_angle_going_well_before_the_position_is_a_topple():
    position = np.concatenate([np.full(150, 3.0), np.full(50, 40.0)])
    orientation = np.concatenate([np.full(80, 5.0), np.full(120, 90.0)])
    verdict = case(
        position_cm=position,
        orientation_deg=orientation,
        achieved_x=np.linspace(0, 0.4, 200),
        reference_x=np.linspace(0, 0.5, 200),
    )
    assert verdict["failure_class"] == "toppled_on_contact"
    assert verdict["failure_frame"] == 80


def test_the_angle_going_with_the_position_is_not_a_topple():
    position = np.concatenate([np.full(100, 3.0), np.full(100, 40.0)])
    orientation = np.concatenate([np.full(95, 5.0), np.full(105, 90.0)])
    verdict = case(
        position_cm=position,
        orientation_deg=orientation,
        achieved_x=np.linspace(0, 0.4, 200),
        reference_x=np.linspace(0, 0.5, 200),
    )
    assert verdict["failure_class"] != "toppled_on_contact"


def test_an_object_on_the_floor_mid_clip_is_a_drop():
    achieved_z = np.concatenate([np.full(100, 0.9), np.full(100, 0.03)])
    verdict = case(
        position_cm=np.concatenate([np.full(100, 3.0), np.full(100, 50.0)]),
        achieved_x=np.linspace(0, 0.4, 200),
        achieved_z=achieved_z,
        reference_x=np.linspace(0, 0.5, 200),
    )
    assert verdict["failure_class"] == "dropped_mid_clip"
    assert verdict["failure_frame"] == 100


def test_a_loss_in_the_last_fifth_is_a_placement_failure():
    position = np.concatenate([np.full(170, 3.0), np.full(30, 40.0)])
    verdict = case(
        position_cm=position,
        achieved_x=np.linspace(0, 0.4, 200),
        reference_x=np.linspace(0, 0.5, 200),
    )
    assert verdict["failure_class"] == "lost_at_placement"
    assert verdict["failure_frame"] == 170


def test_a_clip_tracked_the_whole_way_is_named_rather_than_forced_into_a_class():
    verdict = case(
        position_cm=np.full(200, 6.0),
        achieved_x=np.linspace(0, 0.5, 200),
        reference_x=np.linspace(0, 0.5, 200),
    )
    assert verdict["failure_class"] == "held_but_drifting"
    assert verdict["failure_frame"] is None


def test_a_blip_over_the_gate_is_not_a_loss():
    position = np.full(200, 3.0)
    position[120:124] = 50.0
    verdict = case(
        position_cm=position,
        achieved_x=np.linspace(0, 0.5, 200),
        reference_x=np.linspace(0, 0.5, 200),
    )
    assert verdict["failure_class"] == "held_but_drifting"


def test_a_drop_needs_the_reference_to_stay_up():
    """Both on the floor is the clip ending on the floor, which is not a dropped object."""
    down = np.concatenate([np.full(100, 0.9), np.full(100, 0.03)])
    assert drop_frame(pose(np.zeros(200), down), pose(np.zeros(200), down)) is None
    assert drop_frame(pose(np.zeros(200), down), pose(np.zeros(200), np.full(200, 0.9))) == 100
    assert DROP_Z_M == 0.15


def test_a_clip_with_no_ceiling_on_record_is_still_classed():
    verdict = case(position_cm=np.full(200, 60.0), ceiling=None)
    assert verdict["failure_class"] != "reference_ceiling"


def test_the_summary_adds_headroom_per_class_and_scales_it_to_the_thirty_clip_mean():
    rows = [
        {"tag": "e01", "failure_class": "never_grasped", "auc_headroom": 0.3, "kit_add_auc": 0.1},
        {"tag": "e02", "failure_class": "never_grasped", "auc_headroom": 0.3, "kit_add_auc": 0.2},
        {
            "tag": "e03",
            "failure_class": "held_but_drifting",
            "auc_headroom": 0.1,
            "kit_add_auc": 0.5,
        },
    ]
    summary = summarise(rows)
    assert summary["by_class"]["never_grasped"]["auc_headroom"] == 0.6
    assert summary["by_class"]["never_grasped"]["mean_auc_gain_over_30_if_fixed"] == 0.02
    assert list(summary["by_class"]) == ["never_grasped", "held_but_drifting"]
    assert summary["weak_clips"] == ["e01", "e02"]


def test_closure_reports_whether_any_hand_closes_at_all():
    record = {
        "episodes": {
            "4": {
                "contact_frames": {"left": 30, "right": 0},
                "closure": {
                    "left": {"closure": 0.0, "rule": "none", "palm_depth_m": {"median": 0.0133}},
                    "right": {"closure": 0.0, "rule": "none", "palm_depth_m": {"median": 0.0}},
                },
            }
        }
    }
    closure = closure_of(record, 4)
    assert closure["any_hand_closes"] is False
    assert closure["hands"]["left"]["palm_depth_cm"] == 1.33
    assert closure_of(record, 9) == {"measured": False}


def test_a_clip_that_holds_the_object_is_not_toppled_however_far_it_rotates():
    """The fix to the first version of this rule, which called e26 and e17 topples on angle alone.

    Both hold the object inside 6 cm from the first frame to the last; what they lose is angle.
    """
    verdict = case(
        position_cm=np.full(400, 5.5),
        orientation_deg=np.concatenate([np.full(100, 10.0), np.full(300, 95.0)]),
        achieved_x=np.linspace(0, 0.5, 400),
        reference_x=np.linspace(0, 0.5, 400),
        kit_auc=0.45,
        ceiling=0.87,
    )
    assert verdict["failure_class"] == "held_but_drifting"
    assert verdict["first_topple_frame"] == 100
    assert verdict["loss_event_frame"] is None


def test_the_loss_signal_names_which_rule_fired():
    on_floor = np.concatenate([np.full(100, 0.9), np.full(100, 0.03)])
    by_drop = case(
        position_cm=np.concatenate([np.full(120, 3.0), np.full(80, 50.0)]),
        achieved_x=np.linspace(0, 0.4, 200),
        achieved_z=on_floor,
        reference_x=np.linspace(0, 0.5, 200),
    )
    assert by_drop["loss_signal"] == "z-drop"
    assert by_drop["loss_event_frame"] == 100

    by_position = case(
        position_cm=np.concatenate([np.full(100, 3.0), np.full(100, 50.0)]),
        achieved_x=np.linspace(0, 0.4, 200),
        reference_x=np.linspace(0, 0.5, 200),
    )
    assert by_position["loss_signal"] == "position"
    assert by_position["failure_class"] == "dropped_mid_clip"


def test_the_output_json_and_the_first_wave_record_are_separate_options():
    """`--json` already used dest `record`; a second option on it silently read the output file.

    The collision made the first-wave record resolve to results/failure_taxonomy_wave2.json, whose
    `clips` is a list, and the run died on `.items()` -- after the expensive conversion pass.
    """
    import argparse
    import contextlib
    import io

    from v2d.analyze.taxonomy import main

    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.suppress(SystemExit):
        main(["--help"])
    text = out.getvalue()
    assert "--first-wave-record" in text
    assert "--json" in text
    parser = argparse.ArgumentParser()
    assert isinstance(parser, argparse.ArgumentParser)
