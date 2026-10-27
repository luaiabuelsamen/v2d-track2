"""The window analysis names a mechanism, and a wrong name sends the next run after the wrong fix.

Two of the rules exist because the first version got them wrong on real clips: measuring to the
object's origin rather than its surface said "no palm on the object" for a cane whose handle was in
the hand, and naming an event in a window whose clip had already been lost named the consequence.
"""

from __future__ import annotations

import numpy as np

from v2d.analyze.failure_moments import mechanism, window_of


def record(**over):
    base = {
        "object_speed_m_s": {"in_window_max": 0.0, "clip_max": 1.0, "reference_in_window_max": 0.1},
        "object_height_m": {
            "at_failure": 0.5,
            "window_start": 0.5,
            "support_height": None,
            "reaches_support": None,
        },
        "palm_to_object_cm": {
            "hand": "right",
            "window_start": 5.0,
            "at_failure": 20.0,
            "grew_by": 15.0,
            "was_holding": True,
        },
        "reference_palm_to_reference_object_cm": {
            "window_start": 5.0,
            "at_failure": 4.0,
            "reference_is_holding": True,
        },
        "achieved_off_reference_cm": {
            "window_start": 1.0,
            "at_failure": 30.0,
            "clip_max": 50.0,
            "already_lost_when_the_window_opens": False,
        },
        "body_speed_m_s": {"in_window_mean": 0.01, "moving": False},
        "reference_hand": {
            "finger_closure_rad_window_start": 0.8,
            "finger_closure_rad_at_failure": 0.8,
            "opens_in_window": False,
        },
        "reference_contact": {
            "transitions": [],
            "any_transition": False,
            "hands_active_at_failure": ["right"],
        },
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def test_a_clip_already_off_its_reference_is_not_explained_by_this_window():
    verdict = mechanism(
        record(
            achieved_off_reference_cm={
                "window_start": 40.0,
                "already_lost_when_the_window_opens": True,
            }
        )
    )
    assert verdict.startswith("already lost before this window")


def test_a_rollout_still_on_its_reference_is_reported_as_not_failing_here():
    """e09 and e17: the frames come from the first wave, and we ship a different rollout now."""
    verdict = mechanism(record(achieved_off_reference_cm={"at_failure": 4.4}))
    assert verdict == "the shipping rollout does not fail here (still on its reference)"


def test_the_hand_opening_outranks_everything_that_follows_from_it():
    verdict = mechanism(record(reference_hand={"opens_in_window": True}))
    assert verdict == "the reference hand opens"


def test_an_object_moving_far_faster_than_the_reference_is_flung():
    verdict = mechanism(
        record(object_speed_m_s={"in_window_max": 0.48, "reference_in_window_max": 0.1})
    )
    assert verdict == "the policy's object is flung far faster than the reference moves it"


def test_a_slow_object_is_not_flung_however_small_the_reference_speed():
    verdict = mechanism(
        record(object_speed_m_s={"in_window_max": 0.05, "reference_in_window_max": 0.0})
    )
    assert verdict != "the policy's object is flung far faster than the reference moves it"


def test_a_grip_that_holds_in_the_reference_and_loses_the_object_is_a_slide():
    assert mechanism(record()) == "the object slides out of the grip"


def test_contact_moving_between_hands_outranks_the_slide_it_causes():
    verdict = mechanism(record(reference_contact={"any_transition": True}))
    assert verdict == "contact moves between hands"


def test_the_window_is_clipped_to_the_clip():
    assert window_of(100, 50, window=30) == (20, 51)
    assert window_of(100, 50, window=10) == (40, 51)
    assert window_of(100, 5, window=30) == (0, 6)
    assert window_of(100, 200, window=30) == (69, 100)


def test_the_window_is_the_second_before_the_failure_inclusive():
    start, stop = window_of(1000, 602, window=30)
    assert (start, stop) == (572, 603)
    assert len(np.arange(start, stop)) == 31
