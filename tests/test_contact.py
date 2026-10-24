"""Known-answer tests for contact coverage and its sign test."""

from __future__ import annotations

import pytest

from v2d.score.contact import coverage, ladder, paired, recovery, sign_test

COMPLETE = "Retargeting complete"


def _log(directory, name, left, right, frames):
    path = directory / name
    path.write_text(
        f"left_hand_contact_active: {left}/{frames} frames\n"
        f"right_hand_contact_active: {right}/{frames} frames\n"
        f"{COMPLETE}\n"
    )
    return path


def test_coverage_takes_the_better_hand_not_the_mean(tmp_path):
    """A one-handed task has one hand on the object and one idle; averaging would halve it."""
    _log(tmp_path, "tier2_e00.log", left=400, right=0, frames=800)
    assert coverage(tmp_path) == {0: 0.5}


def test_an_unfinished_clip_is_excluded(tmp_path):
    (tmp_path / "tier2_e01.log").write_text("left_hand_contact_active: 10/100 frames\n")
    assert coverage(tmp_path) == {}


def test_a_clip_with_no_contact_lines_is_excluded(tmp_path):
    (tmp_path / "tier2_e02.log").write_text(f"nothing useful\n{COMPLETE}\n")
    assert coverage(tmp_path) == {}


def test_prefix_separates_tiers_sharing_one_directory(tmp_path):
    _log(tmp_path, "tier1_e00.log", 800, 800, 800)
    _log(tmp_path, "tier2_e00.log", 400, 400, 800)
    assert coverage(tmp_path, "tier1") == {0: 1.0}
    assert coverage(tmp_path, "tier2") == {0: 0.5}


def test_zero_frames_does_not_divide_by_zero(tmp_path):
    _log(tmp_path, "tier2_e03.log", 0, 0, 0)
    assert coverage(tmp_path) == {}


def test_paired_uses_only_shared_episodes():
    assert paired({0: 0.5, 1: 0.4}, {0: 0.3, 2: 0.9}) == {0: pytest.approx(0.2)}


def test_sign_test_on_all_ties_is_p_one():
    assert sign_test({0: 0.0, 1: 0.0}) == {"up": 0, "down": 0, "tied": 2, "p_value": 1.0}


def test_sign_test_reproduces_the_committed_table_vs_floor_result():
    """6 up, 5 down, 19 tied -> p = 1.000, the number quoted in the README."""
    differences = dict.fromkeys(range(6), 0.01)
    differences.update({10 + i: -0.01 for i in range(5)})
    differences.update({20 + i: 0.0 for i in range(19)})
    result = sign_test(differences)
    assert (result["up"], result["down"], result["tied"]) == (6, 5, 19)
    assert result["p_value"] == pytest.approx(1.0)


def test_sign_test_detects_a_one_sided_effect():
    result = sign_test(dict.fromkeys(range(10), 0.05))
    assert result["up"] == 10
    assert result["p_value"] < 0.01


def test_ladder_averages_over_shared_episodes_only():
    series = {"a": {0: 0.4, 1: 0.6, 2: 1.0}, "b": {0: 0.2, 1: 0.4}}
    assert ladder(series) == {"a": 0.5, "b": 0.3}


def test_recovery_is_a_fraction_of_the_available_gap():
    assert recovery(0.4352, 0.3701, 0.4709) == pytest.approx(0.646, abs=0.001)


def test_recovery_of_a_zero_gap_is_undefined_rather_than_infinite():
    assert recovery(0.5, 0.5, 0.5) is None
